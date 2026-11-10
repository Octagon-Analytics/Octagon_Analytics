import pandas as pd
import streamlit as st

from modeling import load_artifacts, predict_matchup

st.set_page_config(page_title="Octagon Analytics", page_icon="🥊", layout="wide")

st.markdown("""
<style>
  .block-container {max-width: 1120px; padding-top: 2.2rem;}
  .hero {padding: 2.2rem 2.4rem; border-radius: 18px; color: white;
         background: linear-gradient(115deg, #111827, #7f1d1d); margin-bottom: 1.5rem;}
  .hero h1 {margin: 0; font-size: 2.3rem;} .hero p {margin: .5rem 0 0; opacity: .86;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Training models from the supplied UFC history...")
def artifacts():
    return load_artifacts()

data = artifacts()
st.markdown("""<div class="hero"><h1>OCTAGON ANALYTICS</h1>
<p>Compare hypothetical UFC matchups using two independently trained models.</p></div>""", unsafe_allow_html=True)

st.caption(f"Historical data through {data['latest_date']:%B %d, %Y}. Predictions use only bouts before the selected matchup date.")

names = data["names"]["name"].tolist()
default_a = names.index("Islam Makhachev") if "Islam Makhachev" in names else 0
default_b = names.index("Ilia Topuria") if "Ilia Topuria" in names else min(1, len(names)-1)

with st.form("matchup"):
    left, right, date_col = st.columns([1, 1, .72])
    fighter_a = left.selectbox("Fighter A", names, index=default_a)
    fighter_b = right.selectbox("Fighter B", names, index=default_b)
    fight_date = date_col.date_input("Hypothetical fight date", value=pd.Timestamp.today().date())
    submitted = st.form_submit_button("Predict matchup", type="primary", use_container_width=True)

if fighter_a == fighter_b:
    st.warning("Select two different fighters to make a prediction.")
    st.stop()

prediction = predict_matchup(fighter_a, fighter_b, fight_date)
st.subheader(f"{fighter_a} vs. {fighter_b}")

tabs = st.tabs(["Logistic regression", "Gradient boosting"])
for tab, model_name in zip(tabs, ["Logistic regression", "Gradient boosting"]):
    probability = prediction["probabilities"][model_name]
    with tab:
        a, center, b = st.columns([1, .55, 1])
        a.metric(fighter_a, f"{probability:.1%}", "win probability")
        center.markdown("### vs.")
        b.metric(fighter_b, f"{1-probability:.1%}", "win probability")
        st.progress(int(probability * 100), text=f"{fighter_a}: {probability:.1%} | {fighter_b}: {1-probability:.1%}")
        if model_name == "Logistic regression":
            st.caption("A regularized linear probability model. It is easier to explain and is a useful benchmark.")
        else:
            st.caption("An ensemble of shallow decision trees. It can model nonlinear relationships and interactions between matchup advantages.")

st.subheader("Pre-fight matchup profile")
labels = {
    "pre_bouts": "UFC bouts", "pre_win_pct": "Smoothed UFC win rate", "last_3_win_pct": "Last 3 fight win rate",
    "age": "Age", "height_inches": "Height (in.)", "reach_inches": "Reach (in.)", "weight_lbs": "Listed weight (lb.)",
    "pre_sig_str_landed_avg": "Avg. significant strikes landed", "pre_td_landed_avg": "Avg. takedowns landed",
    "pre_sub_att_avg": "Avg. submission attempts", "pre_ctrl_seconds_avg": "Avg. control time (sec.)",
}
shown = list(labels)
comparison = pd.DataFrame({
    "Feature": [labels[x] for x in shown], fighter_a: [prediction["a"].get(x) for x in shown],
    fighter_b: [prediction["b"].get(x) for x in shown],
})
for col in [fighter_a, fighter_b]:
    comparison[col] = comparison[col].map(lambda x: "Not available" if pd.isna(x) else f"{x:.2f}")
st.dataframe(comparison, hide_index=True, use_container_width=True)

with st.expander("Model evaluation and methodology"):
    st.write(
        f"Models were evaluated on the most recent 20% of dates (cutoff: {data['cutoff']:%B %d, %Y}); "
        f"training = {data['training_rows']:,} fighter-fight rows and test = {data['test_rows']:,}. "
        "Both sides of a fight are kept in the same chronological split."
    )
    metrics = data["metrics"].copy()
    for col in ["Accuracy", "ROC AUC", "Brier score"]:
        metrics[col] = metrics[col].map(lambda x: f"{x:.3f}")
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    st.markdown("""
    **What goes into a prediction.** The models compare each fighter's prior UFC record, recent form,
    age, listed size, and career averages for striking, takedowns, submissions, reversals, and control.
    Fight-level statistics are shifted back one bout before training, so the outcome being predicted is not used as a feature.

    **Important limits.** The data only describes recorded UFC history and listed biography fields; it does not include training camp,
    injury status, weight-cut quality, style context, or current betting odds. A prediction is an estimate from historical patterns,
    not a betting recommendation or a guarantee.
    """)

st.caption("Built from the project repository's supplied UFC fight and fighter datasets.")
