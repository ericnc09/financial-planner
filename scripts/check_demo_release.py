#!/usr/bin/env python3
"""Offline release gate for the static demo; no research DB or keys required."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from export_demo_snapshot import ROOT, validate_snapshot


def check(dist: Path | None = None) -> None:
    source = ROOT / 'dashboard/public/demo/snapshot.json'
    payload = json.loads(source.read_text())
    validate_snapshot(payload)
    exporter_digest = hashlib.sha256((ROOT / 'scripts/export_demo_snapshot.py').read_bytes()).hexdigest()
    if payload['metadata']['exporter_sha256'] != exporter_digest:
        raise ValueError('The snapshot exporter changed. Review and regenerate the snapshot before publishing.')
    if dist:
        html = (dist / 'index.html').read_text()
        built_payload = dist / 'demo/snapshot.json'
        if built_payload.read_bytes() != source.read_bytes():
            raise ValueError('The built dataset differs from the reviewed snapshot.')
        if not (dist / '_headers').is_file():
            raise ValueError('Cloudflare response headers are missing.')
        scripts = list((dist / 'assets').glob('*.js'))
        if not scripts or 'finAlyse' not in html:
            raise ValueError('The demo build is incomplete.')
        for file in scripts:
            content = file.read_text()
            if re.search(r'/api/|localhost:8000|127\.0\.0\.1:8000|admin_token', content):
                raise ValueError(f'Live API or admin code leaked into the static release: {file.name}')
        forbidden = [path for path in dist.rglob('*') if path.is_file() and (path.suffix in ('.db', '.sqlite', '.pkl') or path.name.startswith('.env'))]
        if forbidden:
            raise ValueError('Private runtime files must not be included in the demo.')
    print(f"Demo verified: {payload['metadata']['snapshot_id']} · {len(payload['tickers'])} companies · historical cutoff {payload['metadata']['as_of']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, help='Also verify this production build directory')
    check(parser.parse_args().dist)
