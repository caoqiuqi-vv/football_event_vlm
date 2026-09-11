"""Inspect MP4 top-level boxes without reading or decoding the media payload."""
from pathlib import Path
import struct


def is_faststart(path: Path) -> bool:
    with path.open('rb') as handle:
        size = path.stat().st_size
        for _ in range(4096):
            start = handle.tell()
            header = handle.read(8)
            if len(header) != 8:
                return False
            length, kind = struct.unpack('>I4s', header)
            minimum = 8
            if length == 1:
                extended = handle.read(8)
                if len(extended) != 8:
                    return False
                length = struct.unpack('>Q', extended)[0]
                minimum = 16
            elif length == 0:
                length = size - start
            if length < minimum or start + length > size:
                return False
            if kind == b'moov':
                return True
            if kind == b'mdat':
                return False
            handle.seek(start + length)
    return False
