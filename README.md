# RouteLens — GeoLife GPS Map Matching

RouteLens is a Streamlit application for comparing GPS observations with the saved Beijing OpenStreetMap driving graph. It supports local road candidates for a single point and probabilistic HMM/Viterbi map matching for a trajectory. The route output is computed from the selected input each time; historical candidate and matched-route CSVs are retained as research artifacts and are not used as predictions.

## Research paper relationship

This is a paper-informed GeoLife implementation, not a numerical reproduction of “Real-Time Dense Map Matching with Naive Hidden Markov Models: Delay versus Accuracy.” It now includes both the existing full-trajectory network-distance HMM baseline and a delayed sliding-window naive-HMM decoder. The sliding model combines GPS-distance, causal heading, and available OSM speed-limit signals in emissions, plus backtrack and midpoint-discrepancy transition features. It does not require shortest-path transitions. Its feature weights are explicit untrained heuristic defaults because the paper does not report its learned L1-SVM weight values. The paper's 0.33%–4.6% classification error is from 1,227 labeled San Francisco/Palo Alto observations and is not a GeoLife result. The Beijing GeoLife sample has no verified road-edge labels, so accuracy/error is blank; coverage and GPS-to-road distance are reported only as separate fit/availability metrics.

The Evaluation page contains the paper reference and current prediction metrics. **Buffer Experiments** runs repeatable nearest-road, full-sequence, and sliding-window comparisons, then saves a results CSV, JSON configuration, and plots under `results/experiments/`. Plots are generated from the measured run. Accuracy plots are omitted without explicit directed-edge labels.

The implementation and measured sample update are summarized in [output/research_implementation_update.md](output/research_implementation_update.md). The older assignment PDF and slide deck in `output/` predate the sliding-window experiment implementation; use the updated source and this handover note for the current state.

## Pipeline

1. Validate WGS84 latitude and longitude values and preserve supplied trajectory segments.
2. Project the saved OSM graph to a local metric CRS and search nearby directed road edges with the project's spatial-index candidate generator.
3. Score GPS-to-road distance with the existing Gaussian emission model (σ = 20 m by default).
4. Score directed network distance against GPS displacement with the existing transition formulation (β = 50 m).
5. For the original baseline, run network-distance Viterbi over connected runs; observations without candidates and directed disconnected components split runs.
6. For buffer experiments, score paper-informed candidate features and use a trailing window of maximum size `T`. On arrival at index `t`, finalize through `t-N`, where `N` is the future-observation buffer. Flush the tail at a segment end or candidate-free gap. The output records the commitment observation and delay in observations and, when timestamps allow, seconds.

The single-point result is a local candidate ranking, not an HMM route. The model is a probabilistic HMM/Viterbi map matcher; it is not a trained neural network.

## Stack

- Python, pandas, NumPy, GeoPandas, Shapely, PyProj, NetworkX, and OSMnx
- Streamlit for the application and Streamlit-Folium/Folium for the interactive map
- Saved GeoLife processed CSVs and `data/road_network/beijing_drive.graphml`

## Repository areas

- `app.py` — Streamlit interface, maps, downloads, evaluation, and buffer experiments
- `src/map_matching/inference.py` — reusable validation, candidate scoring, network transitions, and Viterbi inference
- `src/map_matching/candidate_generation.py` — existing projected-distance road candidate generator
- `src/map_matching/hmm_emission.py`, `hmm_transition.py`, `viterbi.py` — original file-oriented research pipeline
- `src/map_matching/sliding_window.py` — paper-informed feature scores and delayed sliding-window Viterbi
- `src/evaluation/delay_accuracy.py` — baselines, delay/coverage/fit metrics, and measured plots
- `src/evaluation/run_experiment.py` — command-line experiment runner
- `data/processed/user_*/` — processed GeoLife trajectories
- `data/processed/map_matching/` — prior research-stage outputs (not used as live predictions)
- `tests/` — inference integration and sliding-window/experiment checks

## Setup and launch (Windows CMD)

From the project root, activate the existing environment and install the two application packages. Pip leaves already installed packages in place and resolves their dependencies; do not recreate `.venv`.

```bat
cd /d path\to\ML-Projec_work
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads `data/road_network/beijing_drive.graphml` and processed GeoLife CSVs from `data/processed/user_*/`. The GeoLife raw and processed GPS files are deliberately excluded from Git because they contain timestamped location traces. Add the dataset locally before running trajectory examples. The graph is included; credit OpenStreetMap contributors when redistributing its map data. The app does not download a new road network or reprocess raw GeoLife data. The interactive OpenStreetMap tile layer needs internet access. The saved local road graph is also drawn over the map, so prediction geometry remains available when remote tiles cannot load.

## Single-point prediction

Open **Map Matching → Single GPS point**, enter latitude and longitude, and select **Find nearby roads**. The initial coordinates come from the first row of the saved GeoLife example. The result lists nearby edges, road name/type when present, distance in projected metres, and normalized emission scores. The map highlights the GPS point and candidate road geometries.

The optional timestamp is descriptive. GPS accuracy controls the Gaussian emission sigma; it does not alter the observed coordinate. If the search radius contains no road, the application reports that outcome instead of returning a fabricated nearest edge.

## Trajectory map matching

Choose **Trajectory** in Map Matching or open **Trajectory Explorer**. Select a processed GeoLife CSV, upload a CSV, or enter one coordinate per line. CSV uploads can map custom latitude/longitude, timestamp, and segment columns. Timestamp and segment are optional; when a segment column is absent, all observations are treated as one input segment. Rows with missing, nonnumeric, or out-of-range coordinates are rejected with row information.

Choose **Run HMM map matching**. The map displays the original GPS trace, an individual marker for each actual GPS observation, selected road geometries, and trajectory endpoints. Purple markers matched a road; coral markers have no candidate within the search radius. Hover over a point to see its observation number and match details. Use the map layer control to toggle observation points, trace, roads, and matched-road overlays. The app also shows per-point results, segment summaries, run time, and a downloadable CSV derived from that invocation. The CSV contains every input observation, with a `matched` flag and nullable selected-edge fields for unmatched points.

## Evaluation and metric limits

The Evaluation view reports observation count, matched count, coverage, processing time, mean/median GPS-to-selected-road distance, and segment-level coverage. Coverage is the percentage with a selected candidate. GPS-to-road distance is geometric fit to the selected edge. It is not ground-truth accuracy; no ground-truth road labels are evaluated here.

## Buffer experiments

In **Buffer Experiments**, select one or more processed trajectories, window sizes `T`, future buffers `N`, and methods. The app rejects invalid pairs where `N >= T`. The `paper_informed_sliding_window` method commits predictions after seeing `N` future points, while flushing the remaining observations when a segment ends or candidate generation has a gap. The nearest-road baseline has zero observation delay. The full-sequence baseline is an offline method and is labeled as final at the end of the trajectory. Runtime includes candidate preparation shared by all methods plus each method's measured decoding time.

The experiment writes `experiment_results.csv`, `experiment_results.json`, `experiment_config.json`, coverage/delay/runtime PNGs, and—only when directed road-edge labels are available—an accuracy PNG. Exact accuracy is evaluated only from explicit edge labels such as `ground_truth_u/v/key`, `true_u/v/key`, `reference_u/v/key`, or a selected label column in `(u,v,key)` / `u|v|key` form. The reported `invalid_transition_count` counts adjacent chosen edges without a directed path from the prior edge's downstream node to the next edge's upstream node; it is a diagnostic and is not enforced by the naive-HMM decoder.

Run an experiment from the command line:

```bat
python -m src.evaluation.run_experiment --input data/processed/user_000/20081023025304.csv --window-sizes 5 10 15 --buffer-sizes 0 1 2 3 5 7 --baselines nearest_road full_sequence_network_hmm paper_informed_sliding_window
```

The feature weights are heuristic and untrained. Heading is taken from a supplied heading/course column or derived from the previous GPS point only, to avoid future leakage. Speed-limit consistency contributes only where both measured speed and an OSM speed limit are available. Backtrack history uses already committed roads. These choices make the experiment runnable with current GeoLife/OSM data but do not reproduce the paper's learned parameterization.

## Known limits

- GPS noise and the fixed default σ/β parameters affect ranking; the project has not trained these values against labels.
- Roads missing from the saved graph or outside the selected search radius remain unmatched.
- Directed disconnected components can prevent a transition; Viterbi restarts at a new sequence block rather than drawing a false connection.
- The graph covers the saved Beijing driving network, not arbitrary locations.
- Remote map tiles and the optional web font require network access; prediction uses the saved graph locally.

## Checks

Run the focused tests from the project root:

```bat
python -m unittest discover -s tests -v
```
