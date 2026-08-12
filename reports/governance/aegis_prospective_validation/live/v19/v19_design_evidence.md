# V19 Design Evidence

## Label Audit

- Independent episodes: 96,382.
- Population mean frozen utility: -0.2004%.
- Population positive-utility rate: 34.81%.
- V11 clean prevalence: 11.92%.
- V11 clean mean frozen utility: 0.4667%.
- V11 clean positive-utility rate: 100%.
- V11 clean positive-utility recall: 34.24%.
- V11 clean mean MAE: 0.4661%.
- V11 clean mean current-TS protected return: 0.3225%.
- `target_before_stop` prevalence: 42.92%, mean utility 0.1099%.
- `clean_fast_success` prevalence: 40.26%, mean utility 0.1251%.
- Current-TS protected-positive prevalence: 54.52%, mean utility 0.0614%.

The clean label is economically meaningful but incomplete. It should remain a
precision component, not the sole objective.

## Why V18 Mischaracterized Clean Ranking

The exact V18 inner-calibration procedure was reproduced. The raw clean models
had validation AUC 0.7281 LONG and 0.7226 SHORT. Platt slopes were negative
(-0.000021 LONG and -0.000937 SHORT), so calibrated AUC became 0.2719 and
0.2774. This is rank inversion, not ordinary probability calibration.

An exploratory fixed-model comparison found modest signal for several existing
targets, but top-decile selections still had negative mean utility. Therefore
V19 does not replace the clean target with whichever retrospective target
looked best. It preregisters a decomposed economic hypothesis and requires new
future evidence.

## Scientific Disposition

- V18 remains frozen and failed.
- Existing data are design evidence, not V19 confirmation.
- No thresholds were altered to make V18 pass.
- V19 model export is prohibited before fresh validation.
- Current Shadow and Live remain unchanged.
