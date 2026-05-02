from aegis_alpha.env.action_mask import CLOSE, IDLE, LONG, SHORT, ActionMaskConfig, coerce_action, valid_action_mask


def test_flat_requires_cooldown():
    cfg = ActionMaskConfig(min_flat_steps=12)
    mask = valid_action_mask(position=0, hold_steps=0, flat_steps=3, cfg=cfg)
    assert mask[IDLE]
    assert not mask[LONG]
    assert not mask[SHORT]
    assert not mask[CLOSE]
    assert coerce_action(LONG, 0, 0, 3, cfg) == (IDLE, True)


def test_position_requires_hold_for_close():
    cfg = ActionMaskConfig(min_hold_steps=6)
    assert coerce_action(CLOSE, 1, 3, 0, cfg) == (IDLE, True)
    assert coerce_action(CLOSE, 1, 6, 0, cfg) == (CLOSE, False)
