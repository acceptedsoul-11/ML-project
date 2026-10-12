# RouteLens research implementation handover

## What is implemented

RouteLens keeps its original single-point candidate finder and full-trajectory directed-network HMM baseline. It now also has a paper-informed delayed sliding-window naive-HMM decoder and a **Buffer Experiments** Streamlit page. At observation arrival `t`, the decoder uses at most the trailing `T` observations and commits through `t-N`. Remaining observations are flushed at trajectory-segment ends and candidate-free gaps. Per-point output records the point that triggered commitment and delay in observations and seconds when timestamps support it.

The added emissions combine geometric GPS distance, a supplied course or a heading derived only from the preceding observation, and speed-limit consistency when both speed and OSM `maxspeed` exist. Transitions combine a committed-road backtrack heuristic and GPS-versus-road-midpoint distance discrepancy. The paper's learned SVM weights are unavailable in the supplied paper, so the defaults are explicitly untrained heuristics. The sliding decoder does not impose shortest-path transition feasibility; that is measured separately as an invalid-transition diagnostic.

## Measured sample run

The experiment was run on `data/processed/user_000/20081023025304.csv` (908 observations), comparing nearest-road, the existing full-sequence network HMM, and six sliding-window settings (`T` in `{5, 10}`, `N` in `{0, 1, 2}`). Candidate preparation is shared among the methods.

| Method | Coverage | Mean delay | End-to-end runtime | Mean GPS-to-road distance |
|---|---:|---:|---:|---:|
| Nearest road | 96.81% | 0 obs | 15.01 s | 18.48 m |
| Full-sequence network HMM | 96.81% | 90.36 obs | 34.66 s | 21.45 m |
| Sliding, `T=5`, `N=1` | 96.81% | 0.99 obs | 22.20 s | 20.32 m |
| Sliding, `T=5`, `N=2` | 96.81% | 1.96 obs | 21.68 s | 19.89 m |
| Sliding, `T=10`, `N=1` | 96.81% | 0.99 obs | 24.34 s | 20.32 m |
| Sliding, `T=10`, `N=2` | 96.81% | 1.96 obs | 23.85 s | 19.89 m |

The sample has no verified directed-edge ground truth, so accuracy and error were not computed. Coverage is candidate availability, and GPS-to-road distance is geometric fit; neither is classification accuracy. Runtime is a single local run and is not a hardware-independent performance claim. The published 0.33%–4.6% result belongs to the paper's labeled San Francisco/Palo Alto dataset and is not a GeoLife result.

## Reproduce

From the project root on Windows:

```bat
.venv\Scripts\activate.bat
python -m src.evaluation.run_experiment --input data/processed/user_000/20081023025304.csv --window-sizes 5 10 --buffer-sizes 0 1 2 --baselines nearest_road full_sequence_network_hmm paper_informed_sliding_window --output-dir results/experiments/handover_sample
```

The experiment folder contains `experiment_results.csv`, `experiment_results.json`, `experiment_config.json`, measured coverage/delay/runtime charts, and a note explaining why accuracy is unavailable. The app also saves each new run under `results/experiments/` and provides the CSV download.

## Validation

`python -m py_compile app.py src/map_matching/inference.py src/map_matching/sliding_window.py src/evaluation/delay_accuracy.py src/evaluation/run_experiment.py tests/test_sliding_window.py` succeeded. `python -m unittest discover -s tests -v` passed all 11 tests.
