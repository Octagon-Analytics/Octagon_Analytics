# Octagon Analytics - UFC Matchup Predictor

An interactive Streamlit website for comparing potential UFC matchups with two models:

- **Logistic regression:** a regularized, interpretable baseline.
- **Gradient boosting:** shallow boosted trees that can learn nonlinear feature effects.

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

## Method notes

Each completed historical fight becomes two fighter-perspective observations. Features include prior UFC bout count, smoothed win percentage, last-three-fight form, age, height, reach, listed weight, and lagged career averages for striking, takedowns, submissions, reversals, and control time. The app never uses same-fight statistics for the row being predicted. Both perspectives of a fight stay together in the chronological train/test split.

This is a class-project tool for exploring historical UFC data, not betting advice. The data cannot account for camp changes, injuries, weigh-ins, matchup-specific game plans, or other important fight context.
