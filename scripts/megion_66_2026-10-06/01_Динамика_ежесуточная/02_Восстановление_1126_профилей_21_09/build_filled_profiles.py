#!/usr/bin/env python3
"""Build replacement profiles from same-pipe donor profiles; never alters source data."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def number(value: str):
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def render(value):
    return "" if value is None else format(value, ".15g")


def key_from_raw(line: bytes):
    first = line.find(b",")
    second = line.find(b",", first + 1)
    return line[first + 1:second].decode("ascii"), line[:first].decode("ascii")


def split_donors(source: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    current = None
    handle = None
    profiles = rows = 0
    with source.open("rb") as raw:
        raw.readline()
        for line in raw:
            pipe, day = key_from_raw(line)
            marker = (pipe, day)
            if marker != current:
                if handle:
                    handle.close()
                handle = (target / f"{pipe}__{day}.csv").open("wb")
                current = marker; profiles += 1
            handle.write(line); rows += 1
    if handle:
        handle.close()
    return profiles, rows


def read_profile(path: Path):
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 51:
                raise ValueError(f"{path}: {len(row)} fields")
            result[row[18]] = row
    return result


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--donors", type=Path, required=True)
    p.add_argument("--means", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    donor_dir = args.out_dir / "доноры_по_профилям"
    filled_dir = args.out_dir / "заполненные_профили"
    filled_dir.mkdir(parents=True, exist_ok=True)
    donor_profiles, donor_rows = split_donors(args.donors, donor_dir)

    means = {}
    with args.means.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
        if len(header) != 51:
            raise ValueError("Mean profile schema is invalid")
        for row in csv.reader(handle):
            means.setdefault(row[1], {})[row[18]] = row

    with args.plan.open(encoding="utf-8-sig", newline="") as handle:
        plan = list(csv.DictReader(handle))
    manifest = []; total_rows = 0
    for item in plan:
        pipe, day, method = item["ID простого участка"], item["Дата"], item["Метод"]
        result_path = filled_dir / f"{pipe}__{day}.csv"
        if method.startswith("Линейная"):
            before = read_profile(donor_dir / f"{pipe}__{item['Дата до']}.csv")
            after = read_profile(donor_dir / f"{pipe}__{item['Дата после']}.csv")
            if set(before) != set(after):
                raise ValueError(f"Segment grids differ for {pipe}: {item['Дата до']} / {item['Дата после']}")
            weight = float(item["Вес после"])
            rows=[]
            for segment in sorted(before, key=float):
                left, right = before[segment], after[segment]
                row=[]
                for index in range(51):
                    if index == 0: row.append(day)
                    elif index == 1: row.append(pipe)
                    elif index == 18: row.append(segment)
                    else:
                        a, b = number(left[index]), number(right[index])
                        row.append(render(a * (1 - weight) + b * weight) if a is not None and b is not None else render(a if a is not None else b))
                rows.append(row)
        else:
            rows=[]
            for segment in sorted(means[pipe], key=float):
                row=means[pipe][segment].copy(); row[0]=day; row[1]=pipe; row[18]=segment; rows.append(row)
        with result_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(rows)
        total_rows += len(rows)
        manifest.append({"ID простого участка":pipe,"Дата":day,"Метод":method,"Сегментных строк":len(rows),"Файл профиля":result_path.name})
    fields=list(manifest[0])
    with (args.out_dir / "РЕЕСТР_ДОЗАПОЛНЕННЫХ_ПРОФИЛЕЙ.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        w=csv.DictWriter(handle,fieldnames=fields);w.writeheader();w.writerows(manifest)
    report={"status":"PASS","donor_profiles":donor_profiles,"donor_rows":donor_rows,"filled_profiles":len(manifest),"filled_segment_rows":total_rows,"interpolated_profiles":sum(m['Метод'].startswith('Линейная') for m in manifest),"mean_edge_profiles":sum(m['Метод'].startswith('Средний') for m in manifest)}
    (args.out_dir / "ОТЧЕТ_ПОСТРОЕНИЯ_ПРОФИЛЕЙ.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
