#!/usr/bin/env python3
"""Small byte-preservation fixture for the streaming 51-column patcher."""

import hashlib
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    header = b'\xef\xbb\xbfdate,id,' + b','.join(f'f{i}'.encode() for i in range(2, 51)) + b'\n'
    rows = []
    for day, pid in [('2020-01-01', '1750004707'), ('2020-01-02', '1750028246')]:
        values = [day, pid] + [str(i) for i in range(2, 51)]
        rows.append(','.join(values).encode() + b'\n')
    original = header + b''.join(rows)
    (tmp / 'source.csv').write_bytes(original)
    (tmp / 'patch.csv').write_text(
        'id,date,CO2,pH,H2S in Gas Phase,pCO2\n'
        '1750004707,2020-01-01,31.5,6.8,0.73,0.007142857142857143\n',
        encoding='utf-8',
    )
    command = [str(ROOT / 'patch_final_csv_test'), str(tmp / 'patch.csv'),
               str(tmp / 'source.csv'), str(tmp / 'output.csv'), hashlib.sha256(original).hexdigest()]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    final = (tmp / 'output.csv').read_bytes()
    assert final.splitlines()[0] == header.rstrip(b'\n')
    old = rows[0].decode().rstrip('\n').split(',')
    new = final.splitlines()[1].decode().split(',')
    expected = {8: '31.5', 10: '6.8', 38: '0.007142857142857143', 50: '0.73'}
    assert len(new) == 51
    assert all(new[i] == expected.get(i, old[i]) for i in range(51))
    assert final.splitlines(keepends=True)[2] == rows[1]
    assert 'rows=2' in result.stdout and 'matched_id_dates=1' in result.stdout
    check = subprocess.run([str(ROOT / 'verify_final_csv_test'), str(tmp / 'patch.csv'),
                            str(tmp / 'source.csv'), str(tmp / 'output.csv'),
                            hashlib.sha256(original).hexdigest()], check=True, capture_output=True, text=True)
    assert 'PASS rows=2' in check.stdout
    print('PASS: 51 columns, 4 replacements, other bytes unchanged, independent verifier')
