# RouteLens — GeoLife GPS Map Matching

RouteLens is a Streamlit application for comparing GPS observations with a saved Beijing OpenStreetMap driving graph. It supports nearby-road candidates for a single GPS point and full-trajectory HMM/Viterbi map matching. It does not implement the paper's sliding-window or delayed-buffer experiments.

## Authors

Sanvika M Naik, Sairam Charan Nukana

## Research paper relationship

The project is informed by “Real-Time Dense Map Matching with Naive Hidden Markov Models: Delay versus Accuracy,” but it is not a numerical reproduction. RouteLens uses full-trajectory network-distance HMM inference with Gaussian GPS-to-road emissions and directed shortest-path distance transitions. It does not implement the paper's sliding-window decoder, delayed commitment, heading or speed-limit emissions, backtrack or midpoint transition features, or learned SVM weights. Those paper components are background only, not current app features.

The paper's reported 0.33%–4.6% classification error comes from 1,227 labeled San Francisco/Palo Alto observations; it is not a GeoLife result. The Beijing GeoLife sample has no verified road-edge labels. RouteLens reports coverage and GPS-to-road distance as availability and geometric-fit metrics, not ground-truth accuracy.

## Core pipeline

1. Validate WGS84 latitude and longitude values and preserve supplied trajectory segments.
2. Project the saved OSM graph to a local metric CRS and search nearby directed road edges with the spatial-index candidate generator.
3. Score GPS-to-road distance with the Gaussian emission model (σ = 20 m by default).
4. Score directed network distance against GPS displacement with the transition formulation (β = 50 m).
5. Run network-distance Viterbi over connected runs. Observations without candidates and directed disconnected components split runs.

Single-point prediction returns local road candidates. Trajectory prediction uses the existing HMM/Viterbi map matcher; it is not a trained neural network.

## Stack and repository areas

- Python, pandas, NumPy, GeoPandas, Shapely, PyProj, NetworkX, and OSMnx
- Streamlit and Streamlit-Folium/Folium for the app and interactive map
- `app.py` — Streamlit interface, maps, downloads, and evaluation metrics
- `src/map_matching/inference.py` — validation, candidate scoring, network transitions, and HMM/Viterbi inference
- `src/map_matching/candidate_generation.py` — projected-distance road candidate search
- `src/map_matching/hmm_emission.py`, `hmm_transition.py`, and `viterbi.py` — reusable HMM components
- `data/road_network/beijing_drive.graphml` — saved road graph
- `data/processed/user_*/` — local processed GeoLife trajectories
- `tests/` — core inference and map visualization checks

## Setup and launch (Windows CMD)

From the project root, activate the existing environment and install the application requirements:

```bat
cd /d path\to\ML-Projec_work
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads the saved graph and processed GeoLife CSVs from `data/processed/user_*/`. GeoLife raw and processed GPS files are excluded from Git because they contain timestamped location traces. Add the dataset locally before using saved-trajectory examples. The graph is included; credit OpenStreetMap contributors when redistributing its data. The app does not download a new road network or reprocess raw GeoLife data. Interactive OpenStreetMap tiles need internet access; local road geometry is drawn from the saved graph.

## Single-point prediction

Open **Map Matching → Single GPS point**, enter latitude and longitude, and select **Find nearby roads**. The result lists nearby edges, road name/type where available, projected distance in metres, and normalized emission scores. The map highlights the GPS point and candidate road geometry. If no road is inside the search radius, the app reports that rather than inventing a candidate.

## Trajectory map matching

Choose **Trajectory** in Map Matching or open **Trajectory Explorer**. Select a processed GeoLife CSV, upload a CSV, or enter coordinates. Uploads can map custom latitude/longitude, timestamp, and segment columns. Timestamp and segment are optional; without a segment column, all observations form one segment. Rows with missing, nonnumeric, or out-of-range coordinates are rejected with row information.

Select **Run HMM map matching**. The map displays the original GPS trace, an individual marker for every observation, selected road geometry, and trajectory endpoints. Purple markers matched a road; coral markers have no candidate within the search radius. Hover over a point for its observation number and match details. Toggle observation points, trace, roads, and matched-road overlays from the map control. The app shows per-point predictions, segment summaries, coverage, distance and runtime metrics, and a downloadable CSV derived from that prediction. The CSV contains every input observation, a `matched` flag, and nullable selected-edge fields for unmatched points.

## Evaluation and metric limits

The **Evaluation** view reports observation count, matched count, coverage, processing time, mean/median GPS-to-selected-road distance, and segment-level coverage. Coverage is the percentage with a selected candidate. GPS-to-road distance measures geometric fit to the selected edge; it is not ground-truth accuracy, which is not evaluated here.

## Checks

Run the core tests from the project root:

```bat
python -m unittest discover -s tests -v
```
