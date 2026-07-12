# GEN2 F1 Live Canary Spec V2

Status: FROZEN operational capital contract amendment.

Candidate: `gen2-20260711T202935Z`

Changed component:

- `OPERATIONAL_CAPITAL_CONTRACT`

Unchanged components:

- `MODEL_AND_POLICY_FREEZE`
- D3 canonical dataset contract
- TRRM V2
- QMAE V2
- EQM policy
- economic policy
- feature hash
- H12 primary horizon

## Capital Contract V2

- `margin_allocation_fraction`: `0.50`
- `allowed_leverage`: `[15, 20]`
- `default_leverage`: `20`
- `max_concurrent_positions`: `1`
- `max_new_positions_per_30_minutes`: `1`
- `first_arm_max_orders`: `1`
- `equity_stop_fraction`: `0.50`
- `martingale`: `false`
- `averaging_down`: `false`
- `pyramiding`: `false`
- `automatic_compounding`: `true`
- `no_scaling`: `true`

`automatic_compounding` means sizing uses the current available balance at the
time of the candidate opportunity. It does not permit adding to an existing
position, pyramiding, martingale, averaging down, or increasing size after
losses to recover prior losses.

## Sizing Formula

For each eligible opportunity:

1. `allocated_margin = available_balance * 0.50`
2. `target_notional = allocated_margin * selected_leverage`
3. `quantity_raw = target_notional / entry_price`
4. `quantity = floor_to_step_size(quantity_raw)`
5. `actual_notional = quantity * entry_price`
6. `required_isolated_margin = actual_notional / selected_leverage`

The rounded quantity must never exceed the target notional. The 50% margin
allocation is a maximum, not a minimum.

## Leverage Policy

Only 15x and 20x are valid. The default is 20x. If 20x fails liquidation-buffer
safety and 15x passes, the canary selects 15x. Any other leverage is rejected.

Margin mode is isolated only. If isolated margin cannot be confirmed, the
decision is `NO_TRADE / ISOLATED_MARGIN_NOT_CONFIRMED`.

## Equity Floor

At operational revision initialization, capture:

- `initial_canary_equity`
- `equity_floor = initial_canary_equity * 0.50`

The initial equity and floor persist on disk and are not recalculated after
losses or raised after gains during this experiment.

If `current_equity <= equity_floor`:

1. engage kill switch;
2. disarm the canary;
3. block new entries;
4. preserve and reconcile any existing exposure;
5. write `CANARY_EQUITY_FLOOR_REACHED`;
6. require human intervention.

## Stop and Liquidation Safety

For SHORT entries, a protector stop and a timed H12 exit are both required and
must be reduce-only. Horizon exit does not replace the emergency stop.

The pre-order gate must verify:

`entry_price < stop_price_with_buffer < liquidation_price_with_safety_buffer`

If 20x cannot satisfy the buffer, retry 15x without moving the stop. If neither
leverage satisfies the buffer, return `NO_TRADE / LIQUIDATION_BUFFER_INSUFFICIENT`.

Minimum liquidation buffer is controlled by the executable canary contract and
must cover pessimistic slippage, fees, mark-price buffer, rounding error, and an
additional safety margin.

## Arm Token V2

The first live arm token must include:

- candidate id;
- operational revision id;
- initial equity;
- margin fraction `0.50`;
- allowed leverage `[15, 20]`;
- default leverage `20`;
- allowed symbols `ADAUSDT,DOGEUSDT`;
- `max_orders=1`;
- `max_concurrent_positions=1`;
- expiry hours;
- equity stop fraction `0.50`;
- checksum over all fields.

V1 tokens are invalid under Spec V2.

## Enforcement Boundary

This task keeps `REAL_ORDER_SUBMISSION_ENABLED=false`. Dry-runs may read private
account state if credentials are available at runtime, but must submit zero
orders. The canary remains disarmed until a separate human action creates a V2
arm token and a later command waits for the next eligible opportunity.

## Safety

- Phase O new entries must remain paused.
- No old signal may be used for first order.
- No synthetic signal may be fabricated for live order submission.
- Binance is the source of truth for orders, fills, margin mode, positions, and
  open orders.
- GEN2_SYSTEM_FREEZE is the source of truth for the model candidate.
- Paper/live parity records are mandatory for every dry-run or live attempt.
