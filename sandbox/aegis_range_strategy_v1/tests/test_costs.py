from aegis_range_v1.costs import adverse_fill, fee_return, funding_return, gross_return


def test_pure_cost_functions_with_synthetic_events():
    assert adverse_fill(100, "LONG", 2) == 100.02
    assert adverse_fill(100, "SHORT", 2) == 99.98
    assert gross_return("LONG", 100, 110) == 0.1
    assert gross_return("SHORT", 100, 90) == 0.1
    assert fee_return(100, 110) == 0.0005 * 2.1
    events = ((0.001, 100.0), (-0.0005, 102.0))
    assert funding_return("LONG", 100, events) == -funding_return("SHORT", 100, events)
