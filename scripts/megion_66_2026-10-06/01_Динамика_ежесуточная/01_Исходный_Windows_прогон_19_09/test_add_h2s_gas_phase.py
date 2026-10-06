import csv
import json
import tempfile
import unittest
from pathlib import Path

import add_h2s_gas_phase as appender


class H2SGasAppendTests(unittest.TestCase):
    def test_appends_gas_and_preserves_existing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_root, old_base = appender.ROOT, appender.BASE_SERIES
            try:
                appender.ROOT = root
                appender.BASE_SERIES = root / "base.csv"
                with appender.BASE_SERIES.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["id", "date", "H2S in Gas Phase"])
                    writer.writeheader()
                    writer.writerow({"id": "1", "date": "2020-01-01", "H2S in Gas Phase": 10})
                    writer.writerow({"id": "1", "date": "2020-01-02", "H2S in Gas Phase": 0})
                    writer.writerow({"id": "1", "date": "2020-01-03", "H2S in Gas Phase": 30})
                header = ["date", "id"] + [f"c{i}" for i in range(3, 51)]
                source, output = root / "source.csv", root / "output.csv"
                with source.open("w", newline="") as handle:
                    writer = csv.writer(handle); writer.writerow(header)
                    writer.writerow(["2020-01-01", "1"] + ["x"] * 48)
                    writer.writerow(["2020-01-02", "1"] + ["x"] * 48)
                    writer.writerow(["2020-01-03", "1"] + ["x"] * 48)
                report = appender.append(source, output)
                self.assertEqual(report["columns_after"], 51)
                with output.open(newline="", encoding="utf-8-sig") as handle:
                    rows = list(csv.reader(handle))
                self.assertEqual(len(rows[0]), 51)
                self.assertEqual([row[-1] for row in rows[1:]], ["10.0", "20.0", "30.0"])
                self.assertTrue(all(row[0:50] == (["2020-01-01", "1"] + ["x"] * 48) for row in rows[1:2]))
            finally:
                appender.ROOT, appender.BASE_SERIES = old_root, old_base


if __name__ == "__main__":
    unittest.main()
