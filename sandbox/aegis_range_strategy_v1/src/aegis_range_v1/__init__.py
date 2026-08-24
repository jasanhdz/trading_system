"""Pure, deterministic Aegis Range Strategy V1 research engine."""

from .atr import RangeAtr14V1
from .candidates import RangeCandidate, candidate_grid
from .data_adapter import RangeDataAdapter
from .detector import RangeDetectorV1
from .engine import RangeEngineV1
from .levels import RangeLevelsV1
from .lifecycle import RangeLifecycleV1
from .regime import RangeRegimeAdapter
from .safety import RangeSafetyV1
from .signal import RangeSignalV1

__all__ = [
    "RangeAtr14V1",
    "RangeCandidate",
    "RangeDataAdapter",
    "RangeDetectorV1",
    "RangeEngineV1",
    "RangeLevelsV1",
    "RangeLifecycleV1",
    "RangeRegimeAdapter",
    "RangeSafetyV1",
    "RangeSignalV1",
    "candidate_grid",
]
