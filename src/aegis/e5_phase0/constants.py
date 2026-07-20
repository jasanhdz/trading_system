"""Frozen E5 Phase 0 constants copied from authoritative governance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TYPESCRIPT_ROOT = REPOSITORY_ROOT / "binance-futures-bot-ts"
PROTOCOL_ROOT = REPOSITORY_ROOT / "reports/governance/e5_signal_edge_protocol"
PHASE0_REPORT_PATH = PROTOCOL_ROOT / "phase0/e5_phase0_report.json"

EXPERIMENT_ID = "E5_SIGNAL_EDGE_CONTROL_TEST"
PROTOCOL_ID = "E5_PROTOCOL_PREREGISTRATION"
SPECIFICATION_VERSION = "e5-execution-spec-v1"
PHASE0_VERSION = "e5-phase0-v1"
BASE_SEED = 20260718
FUNDING_SCHEMA_VERSION = "e5-funding-history-v1"
FUNDING_PROVIDER = "BINANCE_USDM_FUTURES"
MIN_VALID_REPETITIONS = 9_500
REQUESTED_REPETITIONS = 10_000

CANONICAL_SYMBOLS = (
    "ETHUSDT",
    "BTCUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "SUIUSDT",
    "LTCUSDT",
)
CANONICAL_SYMBOL_SET_HASH = "f6448e67daf1d017e16cc6b331f6494e97e178824474994fff08864303ccd348"


@dataclass(frozen=True)
class AuthorityFile:
    order: int
    name: str
    relative_path: str
    commit: str
    sha256: str


AUTHORITIES = (
    AuthorityFile(1, "original", "e5_protocol_preregistration.md", "b8b86d012c40c4d10f10efb68e5eb9d86d4ac476", "c8057276c93b761b4acca6a6569c8a87468c8b374e34f1bbfffa2b42da3b5770"),
    AuthorityFile(2, "patch_02", "e5_protocol_patch_02.md", "92191db1a7c4135252377f64f51b174f180dcd53", "c668cb28f490ce32524c258791d8d8d58dafb2214939c62871ba43c929bf848e"),
    AuthorityFile(3, "amendment_01", "e5_owner_authorized_amendment_01.md", "943b98a5091c4d9238f754a1e42e63540a4579a6", "c05be85a58e59c3706175f5e2e24ea2343fa63b78e0cc196cdde8ed0faec55a4"),
    AuthorityFile(4, "amendment_02", "e5_owner_authorized_amendment_02.md", "521289606117a478debfca00d2e1fbaa5c2a4301", "b54662ab860e204904ddaf65cc0c1ad046fd5073398045a3d5fc7c36ba418d0f"),
    AuthorityFile(5, "amendment_03", "e5_owner_authorized_amendment_03.md", "5003630ae42a806f79466ec10a4c052ce2a6f28a", "871be087550eb9d632795ded2c8f2633f1e481838198f0ee3ce53b9c8e9a350e"),
    AuthorityFile(6, "amendment_04", "e5_owner_authorized_amendment_04.md", "a76553d15a239735bbb909f96ff3f06426148f50", "a177980633c3280d6eaf6a4a798a6eb623f3692878639894869d2a39f8643774"),
    AuthorityFile(7, "execution_specification", "e5_execution_specification.md", "34441e412f79bc7d12d253040019e857ab5cf2c8", "751b4014f1072e6fd0a49fb3a8820ba60b1b3c556eb94f9fbb7911d70516ae09"),
)

HOLM_TEST_IDS = (
    "A_C1_H12", "A_C1_H48", "A_C1_H96",
    "A_C2_H12", "A_C2_H48", "A_C2_H96",
    "B_SPREAD_H12", "B_SPREAD_H48", "B_SPREAD_H96",
    "C_MONO_H12", "C_MONO_H48", "C_MONO_H96",
)

LABEL_ECONOMICS_REGISTRY = {
    "TRRM": {"field": "tail_event", "type": "binary", "favorable": "lower", "favorable_value": 0},
    "EQM-clean": {"field": "clean_quality", "type": "binary", "favorable": "higher", "favorable_value": 1},
    "EQM-net": {"field": "net_quality_after_costs", "type": "continuous", "favorable": "higher"},
    "QMAE": {"field": "qmae", "type": "continuous", "favorable": "lower"},
}

EXPECTED_LOCKBOX_STATE = {
    "lockbox": "NOT_CONSUMED",
    "consumed_queries": [],
    "budget_remaining": 1,
}
