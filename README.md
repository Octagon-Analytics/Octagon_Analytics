# Octagon Analytics - UFC Matchup Predictor

An interactive Streamlit website for comparing potential UFC matchups with two models:

- **Logistic regression:** a regularized, interpretable baseline.
- **Gradient boosting:** shallow boosted trees that can learn nonlinear feature effects.
- **YOLO Pose footage analysis:** an uploaded clip can be turned into a skeleton-overlay video and exploratory 2D movement summary.

The app uses `merged_stats_n_scorecards.csv` and `fighters_raw.csv` already in this repository. It constructs pre-fight features from previous UFC bouts only, evaluates both models on the newest 20% of event dates, and then refits them on all available historical rows for live predictions.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address shown in the terminal (normally `http://localhost:8501`). First launch trains the models from the CSVs; later interactions reuse them for the active session.

## Deploy with Streamlit Community Cloud

1. Push these files and the two required CSVs to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and choose the repository.
3. Set the main file path to `app.py` and deploy.

No API keys are needed. Keep the CSVs in the repository root because the app reads them beside `app.py`.

## Footage-analysis tab

The first use of the footage tab downloads Ultralytics' small `yolo11n-pose` model automatically. Upload an MP4, MOV, AVI, or MKV clip; the app samples up to the selected duration and returns a downloadable pose-overlay MP4. It also summarizes anonymous tracked subjects' pose confidence, normalized wrist activity, and left elbow/knee angles.

These are exploratory **2D image-plane** metrics. Camera zoom, angle, occlusion, and temporary tracker-ID swaps can change their values. They are intentionally not features in the outcome-prediction models unless a consistent, validated video dataset is later prepared.

## Method notes

Each completed historical fight becomes two fighter-perspective observations. Features include prior UFC bout count, smoothed win percentage, last-three-fight form, age, height, reach, listed weight, and lagged career averages for striking, takedowns, submissions, reversals, and control time. The app never uses same-fight statistics for the row being predicted. Both perspectives of a fight stay together in the chronological train/test split.

This is a class-project tool for exploring historical UFC data, not betting advice. The data cannot account for camp changes, injuries, weigh-ins, matchup-specific game plans, or other important fight context.
