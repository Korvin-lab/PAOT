"""Check all report cells against audit data and render plain-table previews."""
import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

from openpyxl import load_workbook
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def wrapped(draw, text, font, width):
    lines, line = [], ""
    for character in str(text):
        if character == "\n":
            lines.append(line); line = ""
        elif line and draw.textlength(line + character, font=font) > width:
            lines.append(line); line = character
        else:
            line += character
    lines.append(line)
    return lines


def preview(ws, number):
    rows = list(ws.iter_rows())
    widths = [round((ws.column_dimensions[ws.cell(1, c).column_letter].width or 13)*7 + 10)
              for c in range(1, ws.max_column + 1)]
    heights = [math.ceil((ws.row_dimensions[r].height or 15)*4/3)
               for r in range(1, len(rows)+1)]
    image = Image.new("RGB", (sum(widths)+2, sum(heights)+2), "white")
    draw = ImageDraw.Draw(image); font = ImageFont.truetype(FONT, 14)
    y = 0
    for row, height in zip(rows, heights):
        x = 0
        for cell, width in zip(row, widths):
            value = cell.value
            if hasattr(value, "strftime"):
                value = value.strftime("%d.%m.%Y")
            lines = wrapped(draw, "" if value is None else value, font, width-10)
            if cell.alignment.wrap_text:
                assert len(lines)*16 <= height+2, (ws.title, cell.coordinate, lines, height)
            else:
                lines = lines[:1]
            draw.rectangle((x, y, x+width, y+height), outline="#cccccc")
            for index, line in enumerate(lines):
                draw.text((x+5, y+2+16*index), line, font=font, fill="black")
            x += width
        y += height
    image.save(HERE / f"preview_{number}.png")


def main():
    path = HERE / "Мегион_недели_и_проверка_H2S_2026_10_02.xlsx"
    manifest = json.loads((HERE / "FINAL_MANIFEST.json").read_text())
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    wb = load_workbook(path, data_only=True)
    assert wb.sheetnames == ["Итог", "По трубам", "H2S в исходниках"]
    with (HERE / "Сводка_по_трубам_7_дней.csv").open(encoding="utf-8-sig") as stream:
        reader = csv.reader(stream); headers = next(reader); records = list(reader)
    sheet = wb["По трубам"]
    assert [cell.value for cell in sheet[1]] == headers
    actual = list(sheet.iter_rows(min_row=2, values_only=True))
    expected = [tuple([r[0]] + [int(v) for v in r[1:]]) for r in records]
    assert actual == expected
    assert sum(r[1] for r in actual) == manifest["source_pipe_dates"]
    assert sum(r[2] for r in actual) == manifest["pipe_weeks"]
    assert sum(r[5] for r in actual) == manifest["output_rows"]
    evidence = list(wb["H2S в исходниках"].iter_rows(min_row=2, values_only=True))
    assert len(evidence) == 16
    assert sum(r[2] == 0 for r in evidence) == 15
    assert sum(r[2] > 0 for r in evidence) == 1
    assert next(r for r in evidence if r[2] > 0)[0] == "604"
    assert all(ws.freeze_panes is None for ws in wb)
    for number, ws in enumerate(wb, 1):
        for row in ws:
            for cell in row:
                assert cell.data_type != "e", (ws.title, cell.coordinate)
                assert cell.fill.patternType is None
        preview(ws, number)
    output = {"status": "PASS", "sheets": wb.sheetnames,
              "all_pipe_cells_match_csv": True, "h2s_unique_zero_records": 15,
              "h2s_positive_outside_target": 1, "visual_previews": 3,
              "xlsx_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    wb.close()
    (HERE / "workbook_verification.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
