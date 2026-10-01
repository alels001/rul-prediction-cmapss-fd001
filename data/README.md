# Data

The C-MAPSS dataset is published by NASA and is **not** redistributed in this repository.

## Obtaining the data

1. Download the **Turbofan Engine Degradation Simulation Data Set** from the NASA Prognostics Data
   Repository: <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository>
2. Extract the archive.
3. Copy the three FD001 files into this directory:

```
data/
├── train_FD001.txt
├── test_FD001.txt
└── RUL_FD001.txt
```

Only FD001 is used. The other subsets may be present but are ignored.

## File format

`train_FD001.txt` and `test_FD001.txt` are space-separated with no header. Each row is one operating cycle of
one engine, with 26 columns:

| Column | Meaning |
|---|---|
| 1 | engine unit number |
| 2 | cycle index within that engine |
| 3–5 | operational settings 1–3 |
| 6–26 | sensor measurements `s1` … `s21` |

`RUL_FD001.txt` holds one value per line: the true remaining useful life of each test engine at its final
recorded cycle, in the same order as the engine numbers in `test_FD001.txt`.

## Shapes after loading

| | Engines | Rows |
|---|---|---|
| `train_FD001.txt` | 100 | 20,631 |
| `test_FD001.txt` | 100 | 13,096 |
| `RUL_FD001.txt` | 100 | 100 |

A training engine runs to failure, so its remaining life is known at every cycle. A test engine is truncated
before failure and is still running; its true remaining life comes from `RUL_FD001.txt` and is used for
evaluation only.
