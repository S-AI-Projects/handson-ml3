"""Minimal, stdlib-only compressed TFRecord codec (GZIP) for image records.

Implements the TFRecord framing (length, masked CRC32C of length, data,
masked CRC32C of data) and a hand-rolled encoder/decoder for the subset of
``tf.train.Example`` needed here: bytes_list, int64_list and float_list
features.  Files written here are readable by TensorFlow's
``TFRecordDataset(..., compression_type="GZIP")``.
"""
import gzip
import struct

# ---------------------------------------------------------------- CRC32C ---
_POLY = 0x82F63B78
_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ _POLY if _c & 1 else _c >> 1
    _TABLE.append(_c)


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = _TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


# ---------------------------------------------------------------- varints ---
def _varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_varint(buf: bytes, pos: int):
    shift = result = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _ld(field: int, payload: bytes) -> bytes:
    """Length-delimited protobuf field."""
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


# ------------------------------------------------------- Example encoding ---
def encode_example(features: dict) -> bytes:
    """features: name -> bytes | int | list[int] | list[float] | str."""
    entries = b""
    for name, value in features.items():
        if isinstance(value, str):
            value = value.encode("utf-8")
        if isinstance(value, bytes):
            feat = _ld(1, _ld(1, value))  # BytesList
        elif isinstance(value, int):
            feat = _ld(3, _ld(1, _varint(value)))  # Int64List (packed)
        elif value and isinstance(value[0], float):
            feat = _ld(2, _ld(1, struct.pack("<%df" % len(value), *value)))
        else:
            feat = _ld(3, _ld(1, b"".join(_varint(int(v)) for v in value)))
        entry = _ld(1, name.encode("utf-8")) + _ld(2, feat)
        entries += _ld(1, entry)
    return _ld(1, entries)  # Example{features: Features{feature: map}}


def _parse_fields(buf: bytes):
    pos = 0
    while pos < len(buf):
        key, pos = _read_varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 2:
            ln, pos = _read_varint(buf, pos)
            yield field, buf[pos:pos + ln]
            pos += ln
        elif wire == 0:
            val, pos = _read_varint(buf, pos)
            yield field, val
        else:
            raise ValueError("unsupported protobuf wire type %d" % wire)


def decode_example(data: bytes) -> dict:
    out = {}
    for _, features in _parse_fields(data):
        for _, entry in _parse_fields(features):
            name = feature = None
            for f, v in _parse_fields(entry):
                if f == 1:
                    name = v.decode("utf-8")
                elif f == 2:
                    feature = v
            kind, inner = next(_parse_fields(feature))
            payload = next(_parse_fields(inner))[1] if inner else b""
            if kind == 1:
                out[name] = payload
            elif kind == 2:
                out[name] = list(struct.unpack("<%df" % (len(payload) // 4), payload))
            else:
                vals, p = [], 0
                while p < len(payload):
                    v, p = _read_varint(payload, p)
                    vals.append(v - (1 << 64) if v >= 1 << 63 else v)
                out[name] = vals
    return out


# ------------------------------------------------------------- file level ---
def write_tfrecord_gz(path, examples):
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fh:
        for ex in examples:
            data = encode_example(ex)
            length = struct.pack("<Q", len(data))
            fh.write(length)
            fh.write(struct.pack("<I", masked_crc32c(length)))
            fh.write(data)
            fh.write(struct.pack("<I", masked_crc32c(data)))


def read_tfrecord_gz(path):
    out = []
    with gzip.open(path, "rb") as fh:
        while True:
            head = fh.read(8)
            if not head:
                return out
            if len(head) < 8:
                raise ValueError("truncated record header")
            (length,) = struct.unpack("<Q", head)
            (len_crc,) = struct.unpack("<I", fh.read(4))
            if len_crc != masked_crc32c(head):
                raise ValueError("length CRC mismatch")
            data = fh.read(length)
            (data_crc,) = struct.unpack("<I", fh.read(4))
            if len(data) != length or data_crc != masked_crc32c(data):
                raise ValueError("data CRC mismatch")
            out.append(decode_example(data))
