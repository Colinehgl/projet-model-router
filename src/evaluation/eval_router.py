import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from router.router import route

DATA = Path(__file__).resolve().parents[2] / "data" / "test_queries.jsonl"
MAPPING = {"simple": "local", "complex": "api"}

correct, errors = 0, []
with open(DATA, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

for row in rows:
    predicted = route(row["query"])
    expected = MAPPING[row["expected_complexity"]]
    if predicted == expected:
        correct += 1
    else:
        errors.append((row["query"], expected, predicted))

print(f"Précision du routeur par règles : {correct}/{len(rows)} ({100 * correct / len(rows):.0f}%)")
for q, exp, pred in errors:
    print(f"  ✗ attendu={exp} obtenu={pred} | {q}")