import json
import math
from pathlib import Path

from aegis.api import BrainApi
from aegis.domain import DecisionRequest
from aegis.training.experiment import evaluate_authoritative_feature_batch
from aegis.utils import to_primitive


FIXTURE = Path(__file__).parents[1] / "fixtures" / "scientific_parity_golden_v1.json"


def _assert_equivalent(left, right, tolerance: float) -> None:
    if isinstance(left, float) or isinstance(right, float):
        assert math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
        return
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equivalent(left[key], right[key], tolerance)
        return
    if isinstance(left, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_equivalent(left_item, right_item, tolerance)
        return
    assert left == right


def test_experiment_bundle_runtime_and_api_share_authoritative_path(
    snapshot_factory, scenario_bundle_factory, scenario_runtime_factory,
) -> None:
    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    tolerance = float(golden["tolerance"])
    snapshot = snapshot_factory()
    bundle = scenario_bundle_factory(golden["scenario"])
    runtime = scenario_runtime_factory(golden["scenario"], snapshot)
    features = runtime.features.transform(snapshot)

    experiment = evaluate_authoritative_feature_batch(
        bundle,
        features,
        timestamp=snapshot.closed_at,
        config={"protocol": {"friction_fraction": 0.0014}},
        request_id=golden["request_id"],
        decision_cycle_id=golden["decision_cycle_id"],
    )
    request = DecisionRequest(
        golden["request_id"], golden["decision_cycle_id"], "aegis-decision-request-v1",
        runtime.config.contract_version, runtime.config.config_version, snapshot,
    )
    runtime_response = runtime.evaluate(request)
    api_response = BrainApi(runtime).evaluate(request)
    evidence = runtime.evidence.events[-1].payload  # type: ignore[attr-defined]

    _assert_equivalent(to_primitive(experiment.predictions), to_primitive(evidence["predictions"]), tolerance)
    _assert_equivalent(to_primitive(experiment.layers), to_primitive(evidence["layers"]), tolerance)
    _assert_equivalent(to_primitive(experiment.candidates), to_primitive(evidence["candidates"]), tolerance)
    _assert_equivalent(to_primitive(experiment.selection), to_primitive(evidence["selection"]), tolerance)
    assert runtime_response == api_response

    expected = golden["expected"]
    assert runtime_response.status.value == expected["status"]
    assert runtime_response.selected[0].symbol == expected["selected_symbol"]
    assert runtime_response.selected[0].candidate_hash == expected["selected_candidate_hash"]
    assert math.isclose(runtime_response.selected[0].calibrated_score, expected["selected_score"], abs_tol=tolerance)
    actual_scores = {row.symbol: row.calibrated_score for row in experiment.layers.results}
    assert actual_scores.keys() == expected["layer_scores"].keys()
    for symbol, score in expected["layer_scores"].items():
        assert math.isclose(actual_scores[symbol], score, rel_tol=0.0, abs_tol=tolerance)
