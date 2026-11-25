# Examples

`fixture_flows.csv` is **invented** (300 rows, ~55 columns) in the raw-CSV shape the preparation pipeline expects: identifier columns, a constant column, a duplicate column, a mostly-missing column, an infinity, a categorical column, and provider-style label spellings. It is **not** Edge-IIoTset and any model fitted on it says nothing about the benchmark.

```bash
iiot-qelm data check   --csv examples/fixture_flows.csv
iiot-qelm data prepare --csv examples/fixture_flows.csv --out /tmp/prep
iiot-qelm train --data /tmp/prep --config configs/cpu_smoke.yaml --run-dir /tmp/run
```
