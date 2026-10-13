# RouteLens implementation handover

RouteLens currently provides single-point road-candidate lookup and full-trajectory GeoLife map matching over the saved Beijing OSM graph. The trajectory pipeline validates and segments input, generates road candidates, scores Gaussian GPS-to-road emissions and directed network transitions, then runs the existing HMM/Viterbi decoder. The app displays individual GPS observations and matched roads, per-point predictions, segment summaries, coverage and GPS-to-road distance metrics, runtime, and a CSV export.

The project is informed by “Real-Time Dense Map Matching with Naive Hidden Markov Models: Delay versus Accuracy,” but does not implement that paper's sliding-window decoder or delayed-buffer experiments. The paper's reported accuracy is from its separately labeled San Francisco/Palo Alto test drives and must not be reported as a RouteLens GeoLife result. The local Beijing data has no verified road-edge labels, so the app reports coverage and geometric GPS-to-road distance rather than ground-truth accuracy.

## Run the application

From the project root on Windows:

```bat
.venv\Scripts\activate.bat
streamlit run app.py
```

The app reads the saved graph and locally available processed GeoLife CSVs. Raw and processed GPS traces are excluded from Git because they contain timestamped location data. CSV uploads and manual coordinates are also supported.

## Validate core inference

```bat
python -m unittest discover -s tests -v
```

The core inference tests exercise coordinate validation, candidate generation, HMM scoring, transition handling, and Viterbi output. Map tests verify that actual input observations are rendered with matched/unmatched details.
