#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "official" / "task.yml"
CASE_FILES = [
    ROOT / "cases" / "test_cases.txt",
    ROOT / "cases" / "benchmark_cases.txt",
]


def main():
    task_text = TASK.read_text()
    for case_file in CASE_FILES:
        for line_no, line in enumerate(case_file.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fields = dict(part.split(":", 1) for part in line.split(";"))
            nomask_yaml = "True" if fields["nomask"] == "1" else "False"
            needle = (
                f'"seqlen": {fields["seqlen"]}, "bs": {fields["bs"]}, '
                f'"dim": {fields["dim"]}, "hiddendim": {fields["hiddendim"]}, '
                f'"seed": {fields["seed"]}, "nomask": {nomask_yaml}, '
                f'"distribution": "{fields["distribution"]}"'
            )
            if needle not in task_text:
                raise SystemExit(f"{case_file}:{line_no} not found in task.yml: {line}")
    print("case files match official task.yml")


if __name__ == "__main__":
    main()
