"""
Standalone reference implementation for regenerating the AES-CTR keystream
that was superposed as the "tag" on top of the message in the encryption
dataset (data/encryption/superposition_encryption_alpha0.5_R0.5.h5).

This lets you reconstruct the exact ground-truth tag bit sequence for any
document from its enc_conf (KEY, IV, COUNTER) and array length, so you can
validate your own demodulation / joint-detection output against a known
answer instead of just BER against the recovered message.

Requires: h5py, numpy, pycryptodome (pip install pycryptodome)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from Crypto.Cipher import AES

BLOCK = 16  # AES block size in bytes
MOD128 = 1 << 128


def keystream_chunk(key: bytes, iv: bytes, chunk_bits: int, index: int) -> bytes:
    """
    Reproduces src/encryption.py's AESCTRAligned.get_range(..., chunk_bits=..,
    index=..) "chunk mode": slices out the (index-1)'th non-overlapping
    chunk_bits-sized window of the AES-CTR keystream generated from (key, iv).

    index is 1-based. chunk_bits must be a multiple of 128.
    """
    if len(iv) != BLOCK:
        raise ValueError("IV must be exactly 16 bytes (128 bits).")
    if chunk_bits % 128 != 0:
        raise ValueError("chunk_bits must be a multiple of 128.")
    if index <= 0:
        raise ValueError("index must be 1-based and >= 1.")

    bytes_per_chunk = chunk_bits // 8
    start_bytes = bytes_per_chunk * (index - 1)
    block_start = start_bytes // BLOCK
    nblocks = bytes_per_chunk // BLOCK

    ctr0 = (int.from_bytes(iv, "big") + block_start) % MOD128
    cipher = AES.new(key, AES.MODE_CTR, nonce=b"", initial_value=ctr0)
    return cipher.encrypt(b"\x00" * (nblocks * BLOCK))


def keystream_bits(key_hex: str, iv_hex: str, counter: int, n_bits: int) -> np.ndarray:
    """Returns the reconstructed tag as an array of 0/1 bits, length n_bits."""
    key = bytes.fromhex(key_hex)
    iv = bytes.fromhex(iv_hex)
    raw = keystream_chunk(key, iv, chunk_bits=n_bits, index=counter)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits[:n_bits]


def demo(h5_path: str, role: str = "destination", doc_index: int = 0) -> None:
    with h5py.File(h5_path, "r") as f:
        group = f[role]
        enc_conf = json.loads(group["documents/enc_conf_json"][doc_index])
        n_bits = int(group["arrays/r0_lengths"][doc_index])

        print(f"Document #{doc_index} in group '{role}':")
        print(f"  KEY     = {enc_conf['KEY']}")
        print(f"  IV      = {enc_conf['IV']}")
        print(f"  COUNTER = {enc_conf['COUNTER']}")
        print(f"  tag length (bits) = {n_bits}")

        bits = keystream_bits(enc_conf["KEY"], enc_conf["IV"], enc_conf["COUNTER"], n_bits)
        print(f"  reconstructed tag (first 32 bits): {bits[:32]}")
        print()
        print("Use this bit sequence as the ground-truth tag to score your own")
        print("joint-detection / successive-cancellation decoder's tag output,")
        print("in place of relying only on r0/r1 amplitudes.")
        print()
        print("Caveat: enc_conf.COUNTER is a snapshot of the *live* shared counter")
        print("at the moment the receiver logged the frame, not a value verified")
        print("per-frame against the transmitter. If TX and RX drift out of sync")
        print("(e.g. a dropped frame), the reconstructed tag for a given document")
        print("may be off by one or more COUNTER steps from what was actually")
        print("transmitted for that exact frame. Cross-check against BER_tag/")
        print("BER_msg where available before trusting an exact match.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent / "data" / "encryption" / "superposition_encryption_alpha0.5_R0.5.h5"
    )
    demo(path)
