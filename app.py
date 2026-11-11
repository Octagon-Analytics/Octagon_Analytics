import os

import pandas as pd
import streamlit as st

from modeling import load_artifacts, predict_matchup
from video_analysis import analyze_fight_video

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


def render_predictor(data):
    st.caption(f"Historical data through {data['latest_date']:%B %d, %Y}. Predictions use only bouts before the selected matchup date.")
    names = data["names"]["name"].tolist()
    default_a = names.index("Islam Makhachev") if "Islam Makhachev" in names else 0
    default_b = names.index("Ilia Topuria") if "Ilia Topuria" in names else min(1, len(names) - 1)

    with st.form("matchup"):
        left, right, date_col = st.columns([1, 1, .72])
        fighter_a = left.selectbox("Fighter A", names, index=default_a)
        fighter_b = right.selectbox("Fighter B", names, index=default_b)
        fight_date = date_col.date_input("Hypothetical fight date", value=pd.Timestamp.today().date())
        st.form_submit_button("Predict matchup", type="primary", use_container_width=True)

    if fighter_a == fighter_b:
        st.warning("Select two different fighters to make a prediction.")
        return

    prediction = predict_matchup(fighter_a, fighter_b, fight_date)
    st.subheader(f"{fighter_a} vs. {fighter_b}")
    models = st.tabs(["Logistic regression", "Gradient boosting"])
    for tab, model_name in zip(models, ["Logistic regression", "Gradient boosting"]):
        probability = prediction["probabilities"][model_name]
        with tab:
            a, center, b = st.columns([1, .55, 1])
            a.metric(fighter_a, f"{probability:.1%}", "win probability")
            center.markdown("### vs.")
            b.metric(fighter_b, f"{1 - probability:.1%}", "win probability")
            st.progress(int(probability * 100), text=f"{fighter_a}: {probability:.1%} | {fighter_b}: {1-probability:.1%}")
            description = (
                "A regularized linear probability model. It is easier to explain and is a useful benchmark."
                if model_name == "Logistic regression" else
                "An ensemble of shallow decision trees. It can model nonlinear relationships and feature interactions."
            )
            st.caption(description)

    st.subheader("Pre-fight matchup profile")
    labels = {
        "pre_bouts": "UFC bouts", "pre_win_pct": "Smoothed UFC win rate", "last_3_win_pct": "Last 3 fight win rate",
        "age": "Age", "height_inches": "Height (in.)", "reach_inches": "Reach (in.)", "weight_lbs": "Listed weight (lb.)",
        "pre_sig_str_landed_avg": "Avg. significant strikes landed", "pre_td_landed_avg": "Avg. takedowns landed",
        "pre_sub_att_avg": "Avg. submission attempts", "pre_ctrl_seconds_avg": "Avg. control time (sec.)",
    }
    comparison = pd.DataFrame({
        "Feature": list(labels.values()), fighter_a: [prediction["a"].get(x) for x in labels],
        fighter_b: [prediction["b"].get(x) for x in labels],
    })
    for column in [fighter_a, fighter_b]:
        comparison[column] = comparison[column].map(lambda x: "Not available" if pd.isna(x) else f"{x:.2f}")
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    with st.expander("Model evaluation and methodology"):
        st.write(
            f"Models were evaluated on the most recent 20% of dates (cutoff: {data['cutoff']:%B %d, %Y}); "
            f"training = {data['training_rows']:,} fighter-fight rows and test = {data['test_rows']:,}. "
            "Both sides of a fight stay in the same chronological split."
        )
        metrics = data["metrics"].copy()
        for column in ["Accuracy", "ROC AUC", "Brier score"]:
            metrics[column] = metrics[column].map(lambda x: f"{x:.3f}")
        st.dataframe(metrics, hide_index=True, use_container_width=True)
        st.markdown("""
        **What goes into a prediction.** The models compare prior UFC record, recent form, age, listed size,
        and career averages for striking, takedowns, submissions, reversals, and control. Same-fight statistics are shifted
        back one bout before model training.

        **Important limits.** The data does not include camp, injury status, weight-cut quality, matchup-specific strategy,
        or current betting odds. This is an educational estimate, not betting advice or a guarantee.
        """)


def render_footage_analysis():
    st.header("Fight Footage Analysis")
    st.write("Upload a short clip to create a YOLO pose-overlay video and exploratory 2D movement summary.")
    st.info("For best results, use a stable broadcast or side-angle clip with both fighters visible. Processing runs on the server and can take a little time.")
    uploaded_video = st.file_uploader("Upload a fight clip", type=["mp4", "mov", "avi", "mkv"])
    controls = st.columns(3)
    sample_fps = controls[0].select_slider("Pose samples per second", options=[1, 2, 3, 5], value=2)
    max_seconds = controls[1].select_slider("Maximum seconds to analyze", options=[15, 30, 60, 120], value=60)
    confidence = controls[2].select_slider("Detection confidence", options=[0.30, 0.40, 0.50, 0.60], value=0.50)

    if uploaded_video and st.button("Analyze video with YOLO Pose", type="primary"):
        with st.spinner("Loading the pose model and analyzing the clip..."):
            try:
                suffix = os.path.splitext(uploaded_video.name)[1].lower() or ".mp4"
                st.session_state["pose_result"] = analyze_fight_video(
                    uploaded_video.getvalue(), suffix=suffix, sample_fps=sample_fps,
                    max_seconds=max_seconds, confidence=confidence,
                )
                st.session_state["pose_filename"] = f"{os.path.splitext(uploaded_video.name)[0]}_pose_summary.mp4"
            except Exception as error:
                st.error(f"Video analysis could not run: {error}")
                return

    result = st.session_state.get("pose_result")
    if not result:
        return
    summary = result["summary"]
    cards = st.columns(4)
    cards[0].metric("Analyzed", f"{summary['Duration analyzed']:.1f} sec")
    cards[1].metric("Pose samples", summary["Pose samples"])
    cards[2].metric("Detection rate", f"{summary['Detection rate']:.0%}")
    cards[3].metric("People/sample", f"{summary['Mean people detected / sample']:.1f}")
    st.video(result["video_bytes"])
    st.download_button(
        "Download pose-overlay video", result["video_bytes"],
        file_name=st.session_state.get("pose_filename", "pose_summary.mp4"), mime="video/mp4",
    )
    st.subheader("Tracked-subject movement summary")
    table = result["subjects"].copy()
    for column in ["Mean pose confidence", "Mean wrist activity", "Mean left elbow angle", "Mean left knee angle"]:
        if column in table:
            table[column] = table[column].map(lambda x: "Not available" if pd.isna(x) else f"{x:.2f}")
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(
        "Wrist activity is normalized image-plane movement per second; elbow and knee angles are 2D estimates. "
        "Tracked subjects are anonymous and can switch when fighters overlap. These measures are exploratory and are not used in either outcome model."
    )


data = artifacts()
st.markdown("""<div class="hero"><h1>OCTAGON ANALYTICS</h1>
<p>Compare UFC matchups and explore movement in fight footage.</p></div>""", unsafe_allow_html=True)
predictor_tab, footage_tab = st.tabs(["Matchup predictor", "Footage analysis (YOLO Pose)"])
with predictor_tab:
    render_predictor(data)
with footage_tab:
    render_footage_analysis()

st.caption("Built from the project repository's supplied UFC fight and fighter datasets.")
