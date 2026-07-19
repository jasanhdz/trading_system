# E4A_MECHANICAL_EXIT

Preregistered dev-only experiment over the 1,292 frozen E3 SHORT entries. No lockbox or operational path was used.

## Primary B_BASE results

| Policy | Trades | Gross expectancy | Net expectancy | PF | Positive folds | Ambiguous |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 1292 | -0.000034179038 | -0.001534179038 | 0.5370443650824593 | 0 | 0 |
| P1 | 1292 | 0.001024742811 | -0.001175257189 | 0.8642215482892132 | 2 | 0 |
| P2 | 1292 | 0.000887775515 | -0.001127568913 | 0.8564875989884615 | 2 | 0 |
| P3 | 1292 | 0.000066749625 | -0.001635378858 | 0.6191554290877344 | 1 | 459 |
| P4 | 1292 | 0.000266503088 | -0.001332929317 | 0.6046075750233711 | 0 | 804 |

## Classification

`MECHANICAL_EXIT_GROSS_ONLY`

## Next decision

`E4B_CONFIRMATORY_EXIT_HYPOTHESIS_NOT_JUSTIFIED`

## Mandatory questions

- **1_horizon_extension**: `{"delta_net_p1_vs_p0":0.00035892184865177136}`
- **2_fixed_bracket**: `{"delta_net_p2_vs_p0":0.00040661012500865855}`
- **3_break_even**: `{"delta_net_p3_vs_p2":-0.0005078099449991134}`
- **4_full_core**: `{"delta_net_p4_vs_p3":0.0003024495409210679}`
- **5_activation_counts**: `{"BREAK_EVEN_ARMED":1061,"BREAK_EVEN_STOP_UPDATED":305,"BREAK_EVEN_UPDATE_SKIPPED":882,"EXIT":1292,"INTRABAR_AMBIGUITY":811,"PROFIT_PROTECTION_ARMED":1061,"PROFIT_PROTECTION_STOP_UPDATED":1631,"TRAILING_ARMED":286,"TRAILING_STOP_UPDATED":53}`
- **6_fixed_tp_exit_count**: `1`
- **7_initial_stop_exit_count**: `166`
- **8_p4_positive_gross**: `true`
- **9_p4_positive_b_base_net**: `false`
- **10_p4_positive_pessimistic_net**: `false`
- **11_folds_improved_vs_p0**: `2`
- **12_p4_superior_to_p1**: `false`
- **13_capture_ratio_delta**: `-0.4761180790101438`
- **14_giveback_delta**: `0.00048533725311265385`
- **15_tail_and_drawdown**: `{"p0_cvar":-0.01715058370286662,"p0_drawdown":2.0079023126078686,"p4_cvar":-0.02195358974358977,"p4_drawdown":1.8066049002600486}`
- **16_without_best_one_percent_delta**: `0.0001523384751882901`
- **17_symbols_with_positive_delta**: `6`
- **17_symbol_count**: `11`
- **18_intrabar_ambiguous_trades**: `804`
- **19_conservative_delta_positive**: `true`
- **20_maximum_symbol_positive_pnl_share**: `0.12180240722581137`

Optimistic intrabar sensitivity is diagnostic only. The conservative result is the sole promotion input.
