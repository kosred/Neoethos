"""Bounded streaming reads for canonical research artifact arrays."""
import json

def array_items(path, key, raw=False):
    """Stream one top-level named array; never load the 625 MB artifact."""
    decoder = json.JSONDecoder()
    marker = '"' + key + '"'
    with path.open(encoding='utf-8') as f:
        buf = ''
        while marker not in buf:
            chunk = f.read(65536)
            if not chunk: raise ValueError('Missing ' + key)
            buf = buf[-len(marker):] + chunk
        buf = buf[buf.index(marker)+len(marker):]
        while '[' not in buf: buf += f.read(65536)
        buf = buf[buf.index('[')+1:]
        while True:
            buf = buf.lstrip(' \r\n\t,')
            if buf.startswith(']'): return
            try:
                item, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = f.read(65536)
                if not chunk: raise
                buf += chunk
                if len(buf) > 32*1024*1024: raise MemoryError('Single entry exceeds 32 MB budget')
                continue
            yield (item, buf[:end]) if raw else item
            buf = buf[end:]
