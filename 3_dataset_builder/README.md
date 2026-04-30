# analysis\_dataset module

Builds a wide `analysis\_dataset.csv` from:

* `features\_protocol.csv`
* `features\_derived.csv`
* `context\_computed.csv`
* `context\_derived.csv`

It also produces a feature catalog and diagnostic reports.

## Usage

```bash
python -m analysis\_dataset.cli protocol\_with\_analysis\_dataset.yaml --output-dir analysis\_outputs
```

or:

```python
from analysis\_dataset import build\_analysis\_dataset

result = build\_analysis\_dataset("protocol\_with\_analysis\_dataset.yaml", output\_dir="analysis\_outputs")
print(result\["summary"])
```

## Outputs

* `analysis\_dataset.csv` — wide table, one row per date.
* `analysis\_feature\_catalog.csv` — provenance of each generated column.
* `analysis\_summary.json` — compact machine-readable summary.
* `analysis\_report.md` — readable markdown report.
* `analysis\_report/\*.csv` — technical CSV reports.

## Current MVP limitations

* `unit.mode = date` only. `date\_session` is intentionally not implemented yet.
* The module does not calculate new ECG or context features.
* The module does not run feature selection or ML.
* Feature sets are resolved from `protocol.yaml`; direct prefix fallback is disabled by default.



НУЖДАЕТСЯ В ДОРАБОТКЕ!!! 

ПОМИМО ЭТОГО НАДО УПРОСТИТЬ СТРУКТУРУ

