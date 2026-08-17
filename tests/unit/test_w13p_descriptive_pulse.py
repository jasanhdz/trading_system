from aegis.research.w13p_descriptive_pulse import first_barrier


def test_first_barrier_respects_path_order():
    assert first_barrier([(1, 4.0), (2, -6.0), (3, 8.0)], 5.0) == "ADVERSE_FIRST"
    assert first_barrier([(1, 2.0), (2, 7.0), (3, -9.0)], 5.0) == "FAVORABLE_FIRST"
    assert first_barrier([(1, 2.0), (2, -3.0)], 5.0) == "NEITHER"
