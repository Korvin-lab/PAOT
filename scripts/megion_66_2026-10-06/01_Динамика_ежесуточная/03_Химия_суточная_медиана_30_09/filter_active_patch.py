#!/usr/bin/env python3
"""Keep only patch keys corresponding to active final profiles."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
package = ROOT.parent / 'MEGION_WINDOWS_FULL_PE2__PREPARING_2026-09-18'
requested = json.loads((package / 'input/graph_megion_requested_daily_params.json').read_text(encoding='utf-8'))['by_id']
flows = ('Жидкости, м3/сут (дебит)', 'Нефти, т/сут (дебит)', 'Общего газа, тыс.м3/сут (дебит)')
active = {(pid, row['Дата']) for pid, payload in requested.items() for row in payload['daily']
          if any(float(row.get(col) or 0) > 0 for col in flows)}
assert len(active) == 71460
source = ROOT / 'daily_replacements.csv'
output = ROOT / 'daily_replacements_active.csv'
with source.open(newline='', encoding='utf-8') as src, output.open('w', newline='', encoding='utf-8') as dst:
    reader = csv.DictReader(src)
    writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
    writer.writeheader()
    total = kept = 0
    for row in reader:
        total += 1
        if (row['id'], row['date']) in active:
            writer.writerow(row)
            kept += 1
assert total == 88721
print(f'all_patch_keys={total} active_patch_keys={kept} inactive_omitted={total-kept}')
