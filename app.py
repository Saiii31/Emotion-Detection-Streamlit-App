import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from predict import (
    ArtifactsNotFoundError,
    get_emotion_classes,
    get_model_accuracy,
    predict,
)

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMOTION_STYLE = {
    "joy": {"emoji": "😄", "color": "#F5A623"},
    "sadness": {"emoji": "😢", "color": "#2E86DE"},
    "anger": {"emoji": "😠", "color": "#E74C3C"},
    "fear": {"emoji": "😨", "color": "#8E44AD"},
    "love": {"emoji": "❤️", "color": "#E84393"},
    "surprise": {"emoji": "😲", "color": "#F39C12"},
}
DEFAULT_STYLE = {"emoji": "🙂", "color": "#607D8B"}

EXAMPLES = {
    "joy": "I just got the best news ever, I can't stop smiling!",
    "sadness": "I feel so empty and alone since they left.",
    "anger": "I am absolutely furious about how they treated me.",
    "fear": "My hands are shaking, I don't know what's going to happen next.",
    "love": "You mean everything to me and I adore you.",
    "surprise": "Wait, I really didn't expect that at all!",
}


def style_for(emotion: str) -> dict:
    return EMOTION_STYLE.get(emotion, DEFAULT_STYLE)


# ----------------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f8fc 0%, #ffffff 220px);
    }
    .hero {
        padding: 1.75rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #6C63FF 0%, #a78bfa 100%);
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(108, 99, 255, 0.25);
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        opacity: 0.92;
        font-size: 1.02rem;
    }
    .result-card {
        border-radius: 18px;
        padding: 1.75rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 28px rgba(0,0,0,0.15);
        transition: all 0.2s ease-in-out;
    }
    .result-card .emoji {
        font-size: 3.4rem;
        line-height: 1;
    }
    .result-card .label {
        font-size: 1.6rem;
        font-weight: 800;
        text-transform: capitalize;
        margin-top: 0.4rem;
    }
    .result-card .confidence {
        font-size: 0.95rem;
        opacity: 0.92;
        margin-top: 0.2rem;
    }
    .info-card {
        border-radius: 14px;
        padding: 1rem 1.2rem;
        background: #f4f2ff;
        border: 1px solid #e3defc;
        margin-bottom: 0.75rem;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎭 Emotion Detector</h1>
        <p>Type a sentence and instantly see which emotion it expresses —
        powered by TF-IDF + Logistic Regression.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model / handle missing artifacts
# ----------------------------------------------------------------------------
try:
    ACCURACY = get_model_accuracy()
    CLASSES = get_emotion_classes()
    MODEL_READY = True
except ArtifactsNotFoundError:
    MODEL_READY = False
    ACCURACY = None
    CLASSES = list(EXAMPLES.keys())

if not MODEL_READY:
    st.error(
        "No trained model found yet. Run `python src/train.py` from the project "
        "root to train the model and generate the required files in `models/`, "
        "then reload this app."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "text_input" not in st.session_state:
    st.session_state.text_input = ""


def set_example(emotion: str):
    st.session_state.text_input = EXAMPLES[emotion]


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About this model")
    st.markdown(
        f"""
        <div class="info-card">
        <b>Algorithm:</b> Logistic Regression<br>
        <b>Features:</b> TF-IDF vectors<br>
        <b>Test accuracy:</b> {ACCURACY:.1%}<br>
        <b>Classes:</b> {len(CLASSES)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🧹 How the text is cleaned", expanded=False):
        st.markdown(
            """
            1. Lowercase everything
            2. Strip punctuation
            3. Strip digits
            4. Strip non-ASCII characters (emojis)
            5. Remove English stopwords
            6. Vectorize with TF-IDF
            7. Classify with Logistic Regression
            """
        )

    st.subheader("✨ Try an example")
    for emo in CLASSES:
        s = style_for(emo)
        st.button(
            f"{s['emoji']} {emo.capitalize()}",
            key=f"example_{emo}",
            on_click=set_example,
            args=(emo,),
            use_container_width=True,
        )

    if st.session_state.history:
        st.subheader("🕘 Recent predictions")
        for h in reversed(st.session_state.history[-5:]):
            s = style_for(h["emotion"])
            st.caption(f"{s['emoji']} **{h['emotion']}** — \"{h['text'][:40]}\"")

# ----------------------------------------------------------------------------
# Main: single prediction
# ----------------------------------------------------------------------------
tab_single, tab_batch = st.tabs(["📝 Single Prediction", "📂 Batch CSV Prediction"])

with tab_single:
    col_input, col_result = st.columns([1.2, 1], gap="large")

    with col_input:
        st.subheader("Enter your text")
        text = st.text_area(
            "Text to analyze",
            key="text_input",
            height=160,
            placeholder="e.g. I can't believe how amazing today turned out to be!",
            label_visibility="collapsed",
        )
        run = st.button("🔍 Detect Emotion", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Result")
        if run:
            if not text or not text.strip():
                st.warning("Please enter some text first.")
            else:
                result = predict(text)
                if result["emotion"] is None:
                    st.warning(
                        "After cleaning, there wasn't enough meaningful text left to "
                        "classify (try a longer or more descriptive sentence)."
                    )
                else:
                    emotion = result["emotion"]
                    probs = result["probabilities"]
                    confidence = probs[emotion]
                    s = style_for(emotion)

                    st.markdown(
                        f"""
                        <div class="result-card" style="background:{s['color']};">
                            <div class="emoji">{s['emoji']}</div>
                            <div class="label">{emotion}</div>
                            <div class="confidence">{confidence:.1%} confidence</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.session_state.history.append({"text": text, "emotion": emotion})

                    sorted_items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
                    labels = [k.capitalize() for k, _ in sorted_items]
                    values = [v for _, v in sorted_items]
                    colors = [style_for(k)["color"] for k, _ in sorted_items]

                    fig = go.Figure(
                        go.Bar(
                            x=values,
                            y=labels,
                            orientation="h",
                            marker_color=colors,
                            text=[f"{v:.1%}" for v in values],
                            textposition="outside",
                        )
                    )
                    fig.update_layout(
                        height=320,
                        margin=dict(l=10, r=10, t=30, b=10),
                        xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
                        yaxis=dict(autorange="reversed"),
                        title="Confidence across all emotions",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enter a sentence and click **Detect Emotion** to see the result here.")

with tab_batch:
    st.subheader("Upload a CSV for batch prediction")
    st.caption("The file must contain a column named **text**.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            batch_df = None

        if batch_df is not None:
            if "text" not in batch_df.columns:
                st.error("The CSV must have a column named 'text'.")
            else:
                with st.spinner(f"Predicting emotions for {len(batch_df)} rows..."):
                    results = [predict(t) for t in batch_df["text"].astype(str)]

                batch_df["predicted_emotion"] = [r["emotion"] for r in results]
                batch_df["confidence"] = [
                    (r["probabilities"][r["emotion"]] if r["emotion"] else None) for r in results
                ]

                st.success(f"Done! Predicted emotions for {len(batch_df)} rows.")
                st.dataframe(batch_df, use_container_width=True)

                st.subheader("Distribution of predicted emotions")
                counts = batch_df["predicted_emotion"].value_counts()
                fig2 = go.Figure(
                    go.Bar(
                        x=counts.index.str.capitalize(),
                        y=counts.values,
                        marker_color=[style_for(e)["color"] for e in counts.index],
                    )
                )
                fig2.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig2, use_container_width=True)

                csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download results as CSV",
                    data=csv_bytes,
                    file_name="emotion_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.info("Upload a CSV with a 'text' column to run predictions on many rows at once.")

st.markdown("---")
st.caption(
    "Built with Streamlit · TF-IDF + Logistic Regression · "
    f"Test accuracy: {ACCURACY:.1%}" if MODEL_READY else "Built with Streamlit"
)
