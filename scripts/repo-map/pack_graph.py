#!/usr/bin/env python3
"""Split a gzip database into bounded artifacts, or restore it without third-party tools."""
import argparse
import gzip
import shutil
import tempfile
from pathlib import Path

CHUNK_BYTES = 24 * 1024 * 1024
MAX_PARTS = 8


def pack(database, output):
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob('graph.sqlite.gz.part*'):
        old.unlink()
    with tempfile.TemporaryFile() as packed:
        with database.open('rb') as source, gzip.GzipFile(fileobj=packed, mode='wb', mtime=0, compresslevel=9) as target:
            shutil.copyfileobj(source, target)
        packed.seek(0)
        count = 0
        while chunk := packed.read(CHUNK_BYTES):
            if count == MAX_PARTS:
                raise ValueError('Database exceeds the configured eight-artifact limit; increase it explicitly.')
            (output / f'graph.sqlite.gz.part{count:02d}').write_bytes(chunk)
            count += 1
    return count


def restore(parts, database):
    names = sorted(parts.glob('graph.sqlite.gz.part*'))
    if not names or [p.name for p in names] != [f'graph.sqlite.gz.part{i:02d}' for i in range(len(names))]:
        raise ValueError('Missing database parts; download each part into the same folder.')
    with tempfile.TemporaryFile() as compressed:
        for part in names:
            with part.open('rb') as source:
                shutil.copyfileobj(source, compressed)
        compressed.seek(0)
        with gzip.GzipFile(fileobj=compressed, mode='rb') as source, database.open('wb') as target:
            shutil.copyfileobj(source, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('database', type=Path)
    parser.add_argument('parts', type=Path)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    if args.restore:
        restore(args.parts, args.database)
    else:
        print('Database parts:', pack(args.database, args.parts))
