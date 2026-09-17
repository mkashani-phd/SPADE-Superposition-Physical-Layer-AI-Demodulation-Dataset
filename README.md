# Superposition-Coded Physical-Layer Tagging Dataset (Source–Relay–Destination, USRP SDR)

Over-the-air measurements from a 3-node (Source → Relay → Destination) software-defined
radio testbed used to study **physical-layer message authentication via superposition
coding**: instead of appending an authentication tag as a separate frame, the tag is
superposed directly onto the FSK-modulated message waveform at a fraction `alpha` of the
message power, so both are recovered jointly from the same received signal.

This release contains two dataset parts (see [Files](#files)):

1. **`encryption/`** — the tag is an AES-CTR keystream chunk (not a plain HMAC), giving a
   cryptographically-generated, per-transmission tag. `alpha = 0.5`, code rate `R = 0.5`.
   Includes both the **relay**'s and the **destination**'s received signal for the same
   transmissions.
2. **`alpha_sweep/`** — the same superposition scheme with a fixed default message and a
   plain tag (no encryption), swept over `alpha ∈ {0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4}`
   at `R = 0.5`, destination-only. Intended for developing and benchmarking a
   **joint message+tag demodulator** across SNR and superposition power allocation.

## Testbed

Two/three USRP software-defined radios (Source, Relay, Destination) placed in an office
space at variable distance, communicating over the air at ~1.9 GHz using 2-level FSK
modulation. The message (and, superposed on it, the tag) is preceded by a preamble/postamble
for frame synchronization, then LDPC-coded at rate `R`, then FSK-modulated. The receiver
records raw IQ samples for offline post-processing rather than decoding in real time.

For each received frame the two per-symbol soft-decision branches used for message/tag
separation are stored as `r0` and `r1` (the two FSK tone-energy statistics the demodulator
uses to compute LLRs / do joint or successive-cancellation detection — see `src/rx.py` in
the code repository this dataset was generated from).

## Files

> **Note:** the two `.h5` files under `data/` (285 MB and 1.1 GB) exceed GitHub's 100 MB
> per-file limit and are **not tracked in this git repository** (see `.gitignore`). Obtain
> them separately (e.g. Git LFS, Zenodo, IEEE DataPort, or another external host) and place
> them at the paths shown below before running the notebook/scripts.

```
data/
  encryption/
    superposition_encryption_alpha0.5_R0.5.h5   (285 MB)
  alpha_sweep/
    superposition_alpha_sweep_R0.5.h5           (1.1 GB)
scripts/
  export_dataset.py       # how these files were generated from the source MongoDB (for provenance; not needed to use the dataset)
  verify_keystream.py     # regenerate the ground-truth AES-CTR tag for any encryption-dataset document
  requirements.txt
notebooks/
  dataset_tutorial.ipynb  # runnable walkthrough of everything below
LICENSE
README.md
```

### `encryption/superposition_encryption_alpha0.5_R0.5.h5`

Two top-level HDF5 groups, `destination` and `relay`, holding the same transmissions as
seen at each node (1205 and 1208 frames respectively — frame counts differ because a frame
seen by one node isn't always successfully captured/parsed by the other).

### `alpha_sweep/superposition_alpha_sweep_R0.5.h5`

One top-level group per superposition coefficient: `alpha_0.00`, `alpha_0.10`,
`alpha_0.15`, `alpha_0.20`, `alpha_0.25`, `alpha_0.30`, `alpha_0.35`, `alpha_0.40`
(document counts range from 190 to 6619 per group — these were recorded opportunistically,
not with a fixed target count per alpha). Destination only.

### Group layout (same in every group of both files)

```
<group>/documents/id            string   - MongoDB ObjectId of the source record (provenance only)
<group>/documents/SNR           float64  - measured SNR (dB) for this frame
<group>/documents/BER_tag       float64  - bit error rate of the recovered tag vs. ground truth, as computed by the original capture pipeline's decoder
<group>/documents/BER_msg       float64  - bit error rate of the recovered message vs. ground truth, as computed by the original capture pipeline's decoder
<group>/documents/sigma_n2      float64  - estimated noise variance
<group>/documents/config_json   string   - JSON-encoded experiment configuration (see below)
<group>/documents/time_stamp    string   - Unix timestamp string (encryption dataset only)
<group>/documents/enc_conf_json string   - JSON-encoded encryption configuration (encryption dataset only, see below)
<group>/arrays/r0_data          float32  - flattened, concatenated r0 arrays for every document in the group
<group>/arrays/r0_lengths       int64    - length of each document's r0 array (use to slice r0_data back apart)
<group>/arrays/r1_data          float32  - flattened, concatenated r1 arrays
<group>/arrays/r1_lengths       int64    - length of each document's r1 array
```

`r0`/`r1` are variable-length per document in principle, so they are stored flattened
with a parallel `_lengths` array giving each document's slice length — reconstruct with a
cumulative sum of `_lengths` as offsets (`notebooks/dataset_tutorial.ipynb` has a
`get_document()` helper that does this). In practice, within a given file all documents
happen to share the same length (14976 for the encryption dataset, 14592 for the alpha
sweep), but don't rely on that being true in general.

### `config_json` fields (selected)

| Field | Meaning |
|---|---|
| `ALPHA` | Superposition coefficient: fraction of power/amplitude given to the tag relative to the message |
| `MSG_CODE_RATE` | LDPC code rate `R` applied to the message |
| `MAC_LDPC`, `MAC_REP` | Code rate / repetition factor applied to the tag (non-encryption datasets) |
| `MAC_SHA` | Hash algorithm used to compute the plain HMAC tag (`sha256`) — not used in the encryption dataset, where the tag is an AES-CTR keystream instead |
| `SOURCE`, `RELAY`, `DESTINATION` | USRP device identifiers (not IP addresses / no PII) |
| `FREQ`, `RX_RATE`, `TX_RATE`, `TX_SPS` | RF center frequency and sample rates |
| `PREAMBLE`, `POSTAMBLE`, `PREAMBLE_REPEAT` | Frame synchronization sequences |
| `PAYLOAD` | The fixed default test message transmitted in every frame (same across the whole dataset — not secret, it's placeholder text) |
| `SUPERPOSED` | Internal flag from the transmit code path used; `ALPHA` is the field that reflects the actual superposition weight applied |

### `enc_conf_json` fields (encryption dataset only)

| Field | Meaning |
|---|---|
| `KEY` | AES-256 key (hex), used to generate the tag keystream — **published as-is at the authors' request**, see [Reconstructing the ground-truth tag](#reconstructing-the-ground-truth-tag) |
| `IV` | 128-bit CTR initial counter value (hex) |
| `COUNTER` | 1-based chunk index into the AES-CTR keystream used for this frame's tag (see caveat below) |
| `PAYLOAD` | Fixed default test payload used as the plaintext framing for this scheme |
| `MSG_CODERATE` | Code rate used for the message in this scheme |

## Quick start

```python
import h5py, json, numpy as np

f = h5py.File("data/encryption/superposition_encryption_alpha0.5_R0.5.h5", "r")
dest = f["destination"]

print(list(dest.keys()))              # ['arrays', 'documents']
print(dest.attrs["document_count"])   # 1205

lengths = dest["arrays/r0_lengths"][:]
offsets = np.concatenate([[0], np.cumsum(lengths)])
i = 0
r0_i = dest["arrays/r0_data"][offsets[i]:offsets[i+1]]
config_i = json.loads(dest["documents/config_json"][i])
```

See `notebooks/dataset_tutorial.ipynb` for a full runnable walkthrough (loading both
files, building a metadata table with pandas, filtering by SNR, plotting BER vs. SNR
across alpha).

## Reconstructing the ground-truth tag

In the encryption dataset, the tag isn't a fixed HMAC — it's a slice of an AES-CTR
keystream generated from `enc_conf.KEY` and `enc_conf.IV`, taken at the
`enc_conf.COUNTER`-th non-overlapping chunk of length equal to the message (each
transmission advances the counter by one). `scripts/verify_keystream.py` reimplements
this slicing standalone (no dependency on the original lab codebase) so you can
regenerate the exact bit sequence that was superposed as the tag for a given document,
and use it as ground truth to score your own decoder's tag output directly — instead of
only being able to check `BER_tag`/`BER_msg` after the fact.

```bash
pip install -r scripts/requirements.txt
python scripts/verify_keystream.py data/encryption/superposition_encryption_alpha0.5_R0.5.h5
```

**Caveat:** `enc_conf.COUNTER` is a snapshot of the shared counter value at the moment the
*receiver* logged the frame, not a value that was individually verified against what the
*transmitter* used for that exact frame. If transmitter and receiver ever drift out of
sync (e.g. a dropped or corrupted frame between them), the reconstructed tag for a
specific document could be off by one or more counter steps from what was actually
transmitted. Treat exact per-document tag reconstruction as best-effort, and cross-check
against `BER_tag`/`BER_msg` where available before relying on an exact match.

The KEY/IV are published unredacted (rather than being scrubbed for the release) so that
users can independently verify this reconstruction procedure themselves, not only trust
`scripts/verify_keystream.py`'s output.

## Suggested uses

- Benchmarking joint message+tag demodulators (successive cancellation, joint ML/MAP
  detection) against ground truth, across a range of SNR and superposition-power (`alpha`)
  operating points (`alpha_sweep/`).
- Studying the BER/SNR trade-off of superposing a cryptographic tag versus a plain HMAC
  tag on a physical-layer waveform (compare `alpha_sweep/` at `alpha=0` against
  `encryption/`, keeping in mind the encryption dataset only covers `alpha=0.5`).
- Relay-vs-destination channel comparison for the same transmissions (`encryption/`'s
  `relay` and `destination` groups share overlapping transmissions from the same
  experiment run).

## Known limitations

- `alpha_sweep/` groups were collected opportunistically and are not balanced (190 to
  6619 documents per alpha value) or matched in SNR range across alpha values.
- `BER_tag`/`BER_msg` in `alpha_sweep/` were computed by the original capture pipeline's
  own decoder (a specific baseline demodulator), not necessarily an optimal joint or
  successive-cancellation detector. Treat them as a baseline to beat rather than a
  ceiling — recompute your own from `r0`/`r1` with your detector to compare fairly.
- See the [ground-truth tag caveat](#reconstructing-the-ground-truth-tag) above regarding
  TX/RX counter synchronization.

## Provenance

Originally recorded into MongoDB by the lab's experiment-control code and exported to
HDF5 with `scripts/export_dataset.py` (kept here for transparency; you do not need
MongoDB or that script to use this dataset — the `.h5` files are self-contained).

## Citation

If you use this dataset, please cite it — see `CITATION.cff` (also exposed via GitHub's
"Cite this repository" button).

## License

CC BY 4.0 — see `LICENSE`. Replace before publishing if a different license is preferred.

## Contact

mkashani.phd@gmail.com
