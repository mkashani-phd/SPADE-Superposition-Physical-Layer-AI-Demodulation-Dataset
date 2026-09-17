from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pymongo

SCRIPT_DIR = Path(__file__).resolve().parent
CONN_STRING = "mongodb://localhost:27017/"
BATCH_SIZE = 500


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def to_json_string(value: Any) -> str:
    return json.dumps(value, default=json_default, sort_keys=True)


def to_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def to_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def create_resizable(group: h5py.Group, name: str, dtype: Any, compression: str | None = "gzip") -> h5py.Dataset:
    kwargs: dict[str, Any] = {"shape": (0,), "maxshape": (None,), "dtype": dtype, "chunks": True}
    if compression is not None:
        kwargs["compression"] = compression
    return group.create_dataset(name, **kwargs)


def append_1d(dataset: h5py.Dataset, values) -> None:
    if isinstance(values, list):
        values = np.asarray(values, dtype=dataset.dtype)
    if values.size == 0:
        return
    start = dataset.shape[0]
    dataset.resize(start + values.shape[0], axis=0)
    dataset[start:] = values


def export_collection_to_group(
    collection: pymongo.collection.Collection,
    h5group: h5py.Group,
    *,
    array_dtype: np.dtype = np.dtype(np.float32),
    batch_size: int = BATCH_SIZE,
    extra_attrs: dict[str, Any] | None = None,
) -> int:
    """
    Streams a MongoDB collection of r0/r1 documents into an HDF5 group with the
    layout:
      documents/id, documents/SNR, documents/BER_tag, documents/BER_msg,
      documents/sigma_n2, documents/config_json, [documents/time_stamp],
      [documents/enc_conf_json]
      arrays/r0_data + arrays/r0_lengths, arrays/r1_data + arrays/r1_lengths
    """
    string_dtype = h5py.string_dtype(encoding="utf-8")
    docs_group = h5group.create_group("documents")
    arrays_group = h5group.create_group("arrays")

    probe = collection.find_one()
    has_time_stamp = probe is not None and "time_stamp" in probe
    has_enc_conf = probe is not None and "enc_conf" in probe

    datasets = {
        "id": create_resizable(docs_group, "id", string_dtype),
        "BER_tag": create_resizable(docs_group, "BER_tag", np.float64),
        "BER_msg": create_resizable(docs_group, "BER_msg", np.float64),
        "SNR": create_resizable(docs_group, "SNR", np.float64),
        "sigma_n2": create_resizable(docs_group, "sigma_n2", np.float64),
        "config_json": create_resizable(docs_group, "config_json", string_dtype),
        "r0_lengths": create_resizable(arrays_group, "r0_lengths", np.int64),
        "r0_data": create_resizable(arrays_group, "r0_data", array_dtype),
        "r1_lengths": create_resizable(arrays_group, "r1_lengths", np.int64),
        "r1_data": create_resizable(arrays_group, "r1_data", array_dtype),
    }
    if has_time_stamp:
        datasets["time_stamp"] = create_resizable(docs_group, "time_stamp", string_dtype)
    if has_enc_conf:
        datasets["enc_conf_json"] = create_resizable(docs_group, "enc_conf_json", string_dtype)

    total = 0
    batch: list[dict[str, Any]] = []

    def flush(batch_docs: list[dict[str, Any]]) -> None:
        nonlocal total
        if not batch_docs:
            return
        ids, ber_tag, ber_msg, snr, sigma_n2, config_json = [], [], [], [], [], []
        time_stamp, enc_conf_json = [], []
        r0_lengths, r1_lengths, r0_chunks, r1_chunks = [], [], [], []

        for doc in batch_docs:
            r0_arr = np.asarray(doc.get("r0", []), dtype=array_dtype).reshape(-1)
            r1_arr = np.asarray(doc.get("r1", []), dtype=array_dtype).reshape(-1)

            ids.append(to_string(doc.get("_id")))
            ber_tag.append(to_float(doc.get("BER_tag")))
            ber_msg.append(to_float(doc.get("BER_msg")))
            snr.append(to_float(doc.get("SNR")))
            sigma_n2.append(to_float(doc.get("sigma_n2")))
            config_json.append(to_json_string(doc.get("config")))
            r0_lengths.append(int(r0_arr.size))
            r1_lengths.append(int(r1_arr.size))
            r0_chunks.append(r0_arr)
            r1_chunks.append(r1_arr)
            if has_time_stamp:
                time_stamp.append(to_string(doc.get("time_stamp")))
            if has_enc_conf:
                enc_conf_json.append(to_json_string(doc.get("enc_conf")))

        append_1d(datasets["id"], ids)
        append_1d(datasets["BER_tag"], np.asarray(ber_tag, dtype=np.float64))
        append_1d(datasets["BER_msg"], np.asarray(ber_msg, dtype=np.float64))
        append_1d(datasets["SNR"], np.asarray(snr, dtype=np.float64))
        append_1d(datasets["sigma_n2"], np.asarray(sigma_n2, dtype=np.float64))
        append_1d(datasets["config_json"], config_json)
        append_1d(datasets["r0_lengths"], np.asarray(r0_lengths, dtype=np.int64))
        append_1d(datasets["r1_lengths"], np.asarray(r1_lengths, dtype=np.int64))
        append_1d(datasets["r0_data"], np.concatenate(r0_chunks) if r0_chunks else np.asarray([], dtype=array_dtype))
        append_1d(datasets["r1_data"], np.concatenate(r1_chunks) if r1_chunks else np.asarray([], dtype=array_dtype))
        if has_time_stamp:
            append_1d(datasets["time_stamp"], time_stamp)
        if has_enc_conf:
            append_1d(datasets["enc_conf_json"], enc_conf_json)

        total += len(batch_docs)

    cursor = collection.find({}, no_cursor_timeout=True).batch_size(batch_size)
    try:
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= batch_size:
                flush(batch)
                batch.clear()
        flush(batch)
    finally:
        cursor.close()

    h5group.attrs["document_count"] = total
    h5group.attrs["source_db"] = collection.database.name
    h5group.attrs["source_collection"] = collection.name
    h5group.attrs["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    h5group.attrs["layout"] = (
        "documents/* for scalar fields, arrays/r0_data + arrays/r0_lengths and "
        "arrays/r1_data + arrays/r1_lengths for flattened variable-length arrays"
    )
    if extra_attrs:
        for k, v in extra_attrs.items():
            h5group.attrs[k] = v

    return total


def export_encryption_dataset(client: pymongo.MongoClient, output_path: Path) -> None:
    db = client["r0_r1_SC_alpha_0_5_R=0_5_Encryption"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["dataset_name"] = "Superposition-Coded MAC-Tag with AES-CTR Keystream Tag (alpha=0.5, R=0.5)"
        h5f.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        h5f.attrs["alpha"] = 0.5
        h5f.attrs["code_rate_R"] = 0.5

        for role, collection_name in [("destination", "destination, phase_1"), ("relay", "relay, phase_1")]:
            group = h5f.create_group(role)
            collection = db[collection_name]
            n = export_collection_to_group(collection, group, extra_attrs={"role": role})
            print(f"  [{output_path.name}] {role}: {n} documents")


def export_alpha_sweep_dataset(client: pymongo.MongoClient, output_path: Path) -> None:
    alphas = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["dataset_name"] = "Superposition-Coded Fixed Message + Tag, Alpha Sweep (R=0.5, no encryption)"
        h5f.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        h5f.attrs["alphas"] = np.asarray(alphas, dtype=np.float64)
        h5f.attrs["code_rate_R"] = 0.5

        for alpha in alphas:
            db_name = f"r0_r1_SC_alpha_{str(alpha).replace('.', '_')}_R=0_5"
            db = client[db_name]
            collection = db["destination, phase_1"]
            group_name = f"alpha_{alpha:.2f}"
            group = h5f.create_group(group_name)
            n = export_collection_to_group(collection, group, extra_attrs={"alpha": alpha})
            print(f"  [{output_path.name}] {group_name}: {n} documents")


def main() -> None:
    client = pymongo.MongoClient(CONN_STRING)
    try:
        base = SCRIPT_DIR.parent / "data"
        print("Exporting encryption dataset...")
        export_encryption_dataset(client, base / "encryption" / "superposition_encryption_alpha0.5_R0.5.h5")
        print("Exporting alpha-sweep dataset...")
        export_alpha_sweep_dataset(client, base / "alpha_sweep" / "superposition_alpha_sweep_R0.5.h5")
        print("Done.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
