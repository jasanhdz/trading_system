# Historical Compatibility Matrix

Environment: Python 3.12.3, NumPy 2.3.3, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, joblib 1.5.3.

| Serialized global | Pickle | Runtime resolution | Classification | Mechanism |
|---|---|---|---|---|
| `__main__.MedianImputer` | RV2 | isolated `aegis_alpha.tools.gen2_rv2_train.MedianImputer` | `HISTORICAL_CUSTOM_CLASS` | temporary `__main__` binding |
| `aegis_alpha.tools.gen2_rv2_train.MedianImputer` | EQM1 | same isolated historical class | `HISTORICAL_CUSTOM_CLASS` | temporary diagnostic namespace |
| `pandas.Index` | RV2, EQM1 | exact `pandas.Index` identity | `STANDARD_LIBRARY_DEPENDENCY` | normal import with identity check |
| `pandas.StringDtype` | RV2, EQM1 | exact `pandas.StringDtype` identity | `STANDARD_LIBRARY_DEPENDENCY` | normal import with identity check |
| `_loss.CyPinballLoss` | RV2 | exact `sklearn._loss._loss.CyPinballLoss` identity | `LEGACY_SERIALIZATION_MODULE_PATH` | exact `find_class` tuple |
| `_loss.CyHalfBinomialLoss` | EQM1 | exact `sklearn._loss._loss.CyHalfBinomialLoss` identity | `LEGACY_SERIALIZATION_MODULE_PATH` | exact `find_class` tuple |

All entries are replay-only. Wildcards, prefix fallbacks, persistent aliases, production imports, and site-packages changes are forbidden. Ordinary importable NumPy, pandas, and scikit-learn globals observed by `pickletools` remain enumerated in `historical_pickle_dependencies.json`.
