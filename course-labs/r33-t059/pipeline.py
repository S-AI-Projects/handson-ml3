"""R33-T059: order-invariance of a compressed TFRecord image pipeline.

Synthetic CPU smoke fixture only (not the named real dataset). Stdlib only.
Records are framed in the TFRecord layout (length, masked CRC32C, payload,
masked CRC32C) and gzip-compressed. Payload is JSON, not tf.train.Example.
"""
import gzip
import json
import random
import struct

SEED = 59
SIDE = 4
KNOWN_LABELS = (0, 1)
FALLBACK_LABEL = -1
REQUIRED = ("record_id", "group_id", "pixels", "label")


def make_fixture():
    """Six 4x4 images: three all-zero (label 0), three white-diagonal (label 1)."""
    records = []
    for i in range(6):
        label = i % 2
        pixels = [[0] * SIDE for _ in range(SIDE)]
        if label == 1:
            for d in range(SIDE):
                pixels[d][d] = 255
        records.append({"record_id": f"rec-{i:02d}", "group_id": f"g{i // 2}",
                        "pixels": pixels, "label": label})
    return records


# ---- TFRecord codec (stdlib) ----
def _crc_table():
    table = []
    for n in range(256):
        c = n
        for _ in range(8):
            c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
        table.append(c)
    return table


_TABLE = _crc_table()


def crc32c(data):
    c = 0xFFFFFFFF
    for b in data:
        c = _TABLE[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def _masked(data):
    c = crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def write_tfrecord_gz(path, records):
    with gzip.open(path, "wb") as f:
        for rec in records:
            payload = json.dumps(rec, sort_keys=True).encode()
            length = struct.pack("<Q", len(payload))
            f.write(length + struct.pack("<I", _masked(length))
                    + payload + struct.pack("<I", _masked(payload)))


def read_tfrecord_gz(path):
    out = []
    with gzip.open(path, "rb") as f:
        while True:
            head = f.read(8)
            if not head:
                break
            (lcrc,) = struct.unpack("<I", f.read(4))
            if lcrc != _masked(head):
                raise ValueError("corrupt length CRC")
            (n,) = struct.unpack("<Q", head)
            payload = f.read(n)
            (pcrc,) = struct.unpack("<I", f.read(4))
            if pcrc != _masked(payload):
                raise ValueError("corrupt payload CRC")
            out.append(json.loads(payload))
    return out


# ---- validation and model ----
def validate(rec):
    for key in REQUIRED:
        if key not in rec:
            raise ValueError(f"missing field: {key}")
    px = rec["pixels"]
    if len(px) != SIDE or any(len(row) != SIDE for row in px):
        raise ValueError("shape mismatch: expected 4x4")
    return rec


def predict(rec):
    """Fixed rule: any non-zero pixel -> class 1."""
    return 1 if any(v for row in rec["pixels"] for v in row) else 0


def evaluate(records):
    """Align by record_id, map unseen labels to -1, exclude them from accuracy."""
    aligned = sorted((validate(r) for r in records), key=lambda r: r["record_id"])
    rows, correct, scored = [], 0, 0
    for r in aligned:
        label = r["label"] if r["label"] in KNOWN_LABELS else FALLBACK_LABEL
        pred = predict(r)
        rows.append((r["record_id"], label, pred))
        if label != FALLBACK_LABEL:
            scored += 1
            correct += int(pred == label)
    return {"rows": rows, "scored": scored,
            "accuracy": correct / scored if scored else None}


def shuffled(records, seed=SEED):
    out = list(records)
    random.Random(seed).shuffle(out)
    return out
