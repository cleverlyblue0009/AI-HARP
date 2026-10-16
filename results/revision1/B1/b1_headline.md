# B1 -- latency-bounded matched coverage

Derived deadline D_r = (R - v t_r - v^2/2a)/v; t_r = 2.5 s, a = 3.4 m/s^2:
* rural_highway: v = 27.78 m/s, R = 610.4 m (nominal_los), stopping distance 182.9 m -> **D_r = 15.39 s**
* urban_grid: v = 15.28 m/s, R = 479.6 m (nominal_los), stopping distance 72.5 m -> **D_r = 26.65 s**
* urban_nlos: v = 15.28 m/s, R = 101.4 m (nlos_corner), stopping distance 72.5 m -> **D_r = 1.89 s**

## Target variant: fixed

| D | cell | target | oracle (cost) | best fixed: cost | agent cost | regret | margin |
|---|---|---|---|---|---|---|---|
| unbounded | rural_highway/d=2/clear/fog_bank | 0.6410 | dvcast (1.560) | dvcast(n_slots=30): 1.560 | n/a | n/a | n/a |
| unbounded | rural_highway/d=20/clear/fog_bank | 0.8787 | greedy_farthest (0.704) | dvcast(n_slots=30): 0.730 | 0.717 | +1.8% | +1.9% |
| unbounded | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | dvcast(n_slots=30): 1.703 | n/a | n/a | n/a |
| unbounded | urban_nlos/d=20/clear/fog_bank | 0.9261 | counter_based (0.533) | dvcast(n_slots=30): 0.567 | n/a | n/a | n/a |
| D=0.5s | rural_highway/d=2/clear/fog_bank | 0.6410 | slotted_1p (1.755) | slotted_1p(n_slots=3): 1.755 | n/a | n/a | n/a |
| D=0.5s | rural_highway/d=20/clear/fog_bank | 0.8787 | dvcast (0.730) | slotted_1p(n_slots=3): 1.341 | 0.717 | -1.9% | +87.1% |
| D=0.5s | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | slotted_1p(n_slots=3): 3.979 | n/a | n/a | n/a |
| D=0.5s | urban_nlos/d=20/clear/fog_bank | 0.9261 | p_persistence_03 (0.550) | slotted_1p(n_slots=3): 0.832 | n/a | n/a | n/a |
| D=1s | rural_highway/d=2/clear/fog_bank | 0.6410 | slotted_1p (1.654) | slotted_1p(n_slots=8): 1.654 | n/a | n/a | n/a |
| D=1s | rural_highway/d=20/clear/fog_bank | 0.8787 | greedy_farthest (0.704) | slotted_1p(n_slots=8): 0.898 | 0.717 | +1.8% | +25.3% |
| D=1s | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | slotted_1p(n_slots=8): 2.282 | n/a | n/a | n/a |
| D=1s | urban_nlos/d=20/clear/fog_bank | 0.9261 | counter_based (0.533) | slotted_1p(n_slots=8): 0.640 | n/a | n/a | n/a |
| D=1.5s | rural_highway/d=2/clear/fog_bank | 0.6410 | slotted_1p (1.654) | slotted_1p(n_slots=8): 1.654 | n/a | n/a | n/a |
| D=1.5s | rural_highway/d=20/clear/fog_bank | 0.8787 | greedy_farthest (0.704) | slotted_1p(n_slots=8): 0.898 | 0.717 | +1.8% | +25.3% |
| D=1.5s | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | slotted_1p(n_slots=8): 2.282 | n/a | n/a | n/a |
| D=1.5s | urban_nlos/d=20/clear/fog_bank | 0.9261 | counter_based (0.533) | slotted_1p(n_slots=8): 0.640 | n/a | n/a | n/a |
| D=2s | rural_highway/d=2/clear/fog_bank | 0.6410 | slotted_1p (1.654) | slotted_1p(n_slots=20): 1.688 | n/a | n/a | n/a |
| D=2s | rural_highway/d=20/clear/fog_bank | 0.8787 | greedy_farthest (0.704) | slotted_1p(n_slots=20): 0.768 | 0.717 | +1.8% | +7.1% |
| D=2s | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | slotted_1p(n_slots=20): 1.837 | n/a | n/a | n/a |
| D=2s | urban_nlos/d=20/clear/fog_bank | 0.9261 | counter_based (0.533) | slotted_1p(n_slots=20): 0.578 | n/a | n/a | n/a |
| D=derived | rural_highway/d=2/clear/fog_bank | 0.6410 | dvcast (1.560) | slotted_1p(n_slots=20): 1.688 | n/a | n/a | n/a |
| D=derived | rural_highway/d=20/clear/fog_bank | 0.8787 | greedy_farthest (0.704) | slotted_1p(n_slots=20): 0.768 | 0.717 | +1.8% | +7.1% |
| D=derived | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | slotted_1p(n_slots=20): 1.837 | n/a | n/a | n/a |
| D=derived | urban_nlos/d=20/clear/fog_bank | 0.9261 | counter_based (0.533) | slotted_1p(n_slots=20): 0.578 | n/a | n/a | n/a |

## Target variant: rebased

| D | cell | target | oracle (cost) | best fixed: cost | agent cost | regret | margin |
|---|---|---|---|---|---|---|---|
| unbounded | rural_highway/d=2/clear/fog_bank | 0.6410 | dvcast (1.560) | dvcast(n_slots=30): 1.560 | n/a | n/a | n/a |
| unbounded | rural_highway/d=20/clear/fog_bank | 0.8787 | greedy_farthest (0.704) | dvcast(n_slots=30): 0.730 | 0.717 | +1.8% | +1.9% |
| unbounded | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | dvcast(n_slots=30): 1.703 | n/a | n/a | n/a |
| unbounded | urban_nlos/d=20/clear/fog_bank | 0.9261 | counter_based (0.533) | dvcast(n_slots=30): 0.567 | n/a | n/a | n/a |
| D=0.5s | rural_highway/d=2/clear/fog_bank | 0.6242 | slotted_1p (1.755) | slotted_1p(n_slots=3): 1.755 | n/a | n/a | n/a |
| D=0.5s | rural_highway/d=20/clear/fog_bank | 0.8787 | dvcast (0.730) | slotted_1p(n_slots=3): 1.341 | 0.717 | -1.9% | +87.1% |
| D=0.5s | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | slotted_1p(n_slots=3): 3.979 | n/a | n/a | n/a |
| D=0.5s | urban_nlos/d=20/clear/fog_bank | 0.9261 | p_persistence_03 (0.550) | slotted_1p(n_slots=3): 0.832 | n/a | n/a | n/a |
| D=1s | rural_highway/d=2/clear/fog_bank | 0.6242 | dvcast (1.640) | slotted_1p(n_slots=8): 1.654 | n/a | n/a | n/a |
| D=1s | rural_highway/d=20/clear/fog_bank | 0.8787 | greedy_farthest (0.704) | slotted_1p(n_slots=8): 0.898 | 0.717 | +1.8% | +25.3% |
| D=1s | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | slotted_1p(n_slots=8): 2.282 | n/a | n/a | n/a |
| D=1s | urban_nlos/d=20/clear/fog_bank | 0.9261 | counter_based (0.533) | slotted_1p(n_slots=8): 0.640 | n/a | n/a | n/a |
| D=1.5s | rural_highway/d=2/clear/fog_bank | 0.6242 | dvcast (1.640) | slotted_1p(n_slots=12): 1.794 | n/a | n/a | n/a |
| D=1.5s | rural_highway/d=20/clear/fog_bank | 0.8787 | greedy_farthest (0.704) | slotted_1p(n_slots=12): 0.796 | 0.717 | +1.8% | +11.0% |
| D=1.5s | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | slotted_1p(n_slots=12): 2.062 | n/a | n/a | n/a |
| D=1.5s | urban_nlos/d=20/clear/fog_bank | 0.9261 | counter_based (0.533) | slotted_1p(n_slots=12): 0.606 | n/a | n/a | n/a |
| D=2s | rural_highway/d=2/clear/fog_bank | 0.6242 | dvcast (1.640) | slotted_1p(n_slots=20): 1.688 | n/a | n/a | n/a |
| D=2s | rural_highway/d=20/clear/fog_bank | 0.8787 | greedy_farthest (0.704) | slotted_1p(n_slots=20): 0.768 | 0.717 | +1.8% | +7.1% |
| D=2s | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | slotted_1p(n_slots=20): 1.837 | n/a | n/a | n/a |
| D=2s | urban_nlos/d=20/clear/fog_bank | 0.9261 | counter_based (0.533) | slotted_1p(n_slots=20): 0.578 | n/a | n/a | n/a |
| D=derived | rural_highway/d=2/clear/fog_bank | 0.6410 | dvcast (1.560) | slotted_1p(n_slots=20): 1.688 | n/a | n/a | n/a |
| D=derived | rural_highway/d=20/clear/fog_bank | 0.8787 | greedy_farthest (0.704) | slotted_1p(n_slots=20): 0.768 | 0.717 | +1.8% | +7.1% |
| D=derived | rural_highway/d=80/clear/fog_bank | 0.8695 | weighted_p (1.512) | slotted_1p(n_slots=20): 1.837 | n/a | n/a | n/a |
| D=derived | urban_nlos/d=20/clear/fog_bank | 0.9261 | counter_based (0.533) | slotted_1p(n_slots=20): 0.578 | n/a | n/a | n/a |
