# E4 Split And Episode Audit

| Split | Rows | Symbol-hour episodes |
|---|---:|---:|
| TRAIN | 284,988 | 11,880 |
| CALIBRATION | 126,720 | 5,280 |
| VALIDATION | 183,686 | 7,655 |
| FINAL_HOLDOUT | 158,400 feature-only | sealed |

There are 595,394 labeled rows but only 24,815 independent symbol-hour
episodes and 2,256 UTC-hour temporal blocks. LONG and SHORT rows from the same
market state share group identity. Twelve 5-minute evaluations in one hour are
not treated as twelve independent market episodes.

One-day embargo intervals separate TRAIN, CALIBRATION, VALIDATION and
FINAL_HOLDOUT, exceeding the 60-minute outcome horizon. Bootstrap intervals use
UTC-hour blocks, not individual rows.
