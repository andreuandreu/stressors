## Files and dependencies

[`hazard-learning_analytical.py`](hazard-learning_analytical.py) implements a deterministic, expected-value approximation of [`hazard-learning_dynamics.py`](hazard-learning_dynamics.py).

The model uses:

- [`hazard-learning_analytical.py`](hazard-learning_analytical.py): analytical model;
- [`hazard-learning_dynamics.py`](hazard-learning_dynamics.py): stochastic agent-based model and network builder;
- [`config_LearnHaz.py`](config_LearnHaz.py): shared parameters;
- [`plot_hazard_learning_comparison.py`](plot_hazard_learning_comparison.py): comparison plot for both models.

The analytical script imports NumPy and dynamically reuses `build_social_network` from the stochastic script. This ensures that both models can construct the same network type for the same seed.


## Output fields

`run_simulation()` returns a dictionary containing:

| Field | Definition |
|---|---|
| `time_history` | Monthly time values. |
| `hazard_frequency_history` | $f(t)$, hazard events per agent per month. |
| `expected_events_history` | $Nf(t)$, expected total population events per month. |
| `mean_memory_history` | $\overline{M}(t)$. |
| `mean_knowledge_history` | $\overline{K}(t)$. |
| `known_fraction_history` | Fraction with $L_i(t)\geq L_{\max}$. |
| `practice_use_fraction_history` | $\overline{U}(t)$, expected practice-use fraction. |
| `expected_event_severity` | $\bar{S}=\mathbb{E}[|X|]$. |
| `network` | Adjacency dictionary of the social network. |

When invoked from the command line, the dictionary is written as a pickle file. The default output is:

```text
./data/hazard_learning_analytical_results.pkl
```

### Parameters
  
The parameters are defined in [`config_LearnHaz.py`](config_LearnHaz.py).

## Running the model

From the repository root:

```bash
python hazard-learning_analytical.py
```

A shorter test run can be executed with:

```bash
python hazard-learning_analytical.py \
  --agents 100 \
  --months 60 \
  --network small_world \
  --seed 123 \
  --output ./data/hazard_learning_analytical_test.pkl
```

The available network choices are:

```text
small_world
erdos_renyi
complete
```

## Comparing with the stochastic simulation

The comparison script runs both models with the same population size, duration, network type, and seed:

```bash
python plot_hazard_learning_comparison.py
```

For an explicit comparison:

```bash
python plot_hazard_learning_comparison.py \
  --agents 250 \
  --months 240 \
  --network small_world \
  --seed 123 \
  --output ./plots/hazard_learning_comparison.png
```

The resulting figure compares:

1. total hazard events;
2. mean severity memory;
3. mean practice knowledge;
4. practice-use fraction.

The blue curve is the stochastic agent simulation. The orange curve is the analytical expected value. Differences are expected because the simulation observes realized random events and decisions, while the analytical model propagates probabilities and expectations.



For each month $t = 0,1,\ldots,T$:

1. **Calculate hazard frequency.**
   
   Compute $f(t)=f_0e^{rt}$.

2. **Calculate expected impact.**
   
   Compute the expected absolute Gumbel severity $\bar{S}$ once during initialization, then calculate $I(t)=f(t)\bar{S}$.

3. **Update severity memory.**
   
   Apply $M_i(t+1)=e^{-\delta_M}M_i(t)+I(t)$.

4. **Calculate neighbor exposure.**
   
   For every agent, calculate $P_i(t)=1-\prod_{j\in\mathcal{N}_i}(1-U_j(t))$.

5. **Calculate learning and forgetting outcomes.**
   
   Compute the one-exposure progress $L_i^{+}(t)$ and the no-exposure progress $L_i^{-}(t)$.

6. **Update learning progress.**
   
   Take the expected mixture:
   
   $$
   L_i(t+1)=P_i(t)L_i^{+}(t)+(1-P_i(t))L_i^{-}(t).
   $$

7. **Normalize knowledge.**
   
   Compute $K_i(t)=\min(1,L_i(t)/L_{\max})$.

8. **Calculate hazard response.**
   
   Compute $R_i(t)=1-e^{-M_i(t)/M_0}$.

9. **Apply cost penalty and update use probability.**
   
   Compute $U_i(t)=\operatorname{clip}(K_i(t)R_i(t)e^{-c/s_c},0,1)$.

10. **Record output histories.**
    
    Store time, hazard frequency, expected total events, mean memory, mean knowledge, fully-known fraction, and mean practice-use probability.
