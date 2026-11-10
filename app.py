import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix

THRESHOLD = 0.3
APP_DIR = Path(__file__).resolve().parent


def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\S+", " url ", t)
    t = re.sub(r"\d+", " num ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extra_features(texts):
    return csr_matrix(np.c_[
        texts.str.len() / 160,
        texts.str.count(r"\d") / 20,
        texts.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1)),
        texts.str.contains(r"\d{9,}").astype(int),
        texts.str.contains(r"£|å£|\d+p\b|pobox", case=False).astype(int),
    ])


@st.cache_resource
def load_artifacts():
    return (
        joblib.load(APP_DIR / "spam_model.pkl"),
        joblib.load(APP_DIR / "tfidf.pkl"),
    )


st.set_page_config(
    page_title="Kind Inbox | SMS checker",
    page_icon="💌",
    layout="centered",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #41354f;
        --muted: #81768d;
        --pink: #f27e9d;
        --purple: #9272d5;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 12% 4%, #ffe6ed 0, transparent 34%),
            radial-gradient(ellipse at 91% 19%, #eee8ff 0, transparent 31%),
            #fffaf8;
    }

    [data-testid="stHeader"] { background: transparent; }
    .block-container {
        max-width: 790px;
        padding: 3rem 1.2rem 2.5rem;
    }

    h1, h2, h3, p, label, button, textarea, .stCaption {
        font-family: 'Trebuchet MS', Arial, sans-serif;
    }

    .hero { text-align: center; margin: 0 auto 2rem; }
    .brand-pill {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .42rem .85rem;
        border: 1px solid #f4dce6;
        border-radius: 999px;
        background: rgba(255,255,255,.82);
        color: #a45b78;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .04em;
    }
    .hero h1 {
        margin: .85rem 0 .45rem;
        color: var(--ink);
        font-family: 'Trebuchet MS', Arial, sans-serif;
        font-size: clamp(2.2rem, 7vw, 3.25rem);
        font-weight: 900;
        letter-spacing: -.045em;
        line-height: 1.12;
    }
    .hero h1 span {
        color: var(--pink);
        background: linear-gradient(100deg, #ed789a, #9b79dc);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        margin: 0 auto;
        max-width: 470px;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
    }
    .helper-heading {
        margin: .2rem 0 .5rem;
        color: #766b81;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .04em;
    }
    [data-testid="stForm"] {
        padding: 1.15rem 1.2rem 1rem;
        border: 1px solid #f1e4eb;
        border-radius: 24px;
        background: rgba(255,255,255,.88);
        box-shadow: 0 18px 55px rgba(90, 62, 91, .08);
    }
    [data-testid="stTextArea"] label {
        color: var(--ink);
        font-weight: 700;
    }
    [data-testid="stTextArea"] textarea {
        border: 1px solid #ebdfea;
        border-radius: 16px;
        background: #fffdfd;
        color: var(--ink);
        line-height: 1.6;
        transition: border-color .2s, box-shadow .2s;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #d7b8e9;
        box-shadow: 0 0 0 3px rgba(184, 145, 215, .14);
    }
    [data-testid="stFormSubmitButton"] { width: 100%; }
    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 3rem;
        border: 0;
        border-radius: 14px;
        background: linear-gradient(100deg, #f18aa5, #a282dc);
        color: white;
        font-size: 1rem;
        font-weight: 800;
        box-shadow: 0 8px 20px rgba(190, 123, 177, .23);
        transition: transform .15s, box-shadow .15s;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        border: 0;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 11px 24px rgba(190, 123, 177, .3);
    }
    [data-testid="stFormSubmitButton"] button:active { transform: translateY(0); }
    [data-testid="stButton"] button {
        border: 1px solid #eadfec;
        border-radius: 12px;
        background: rgba(255,255,255,.8);
        color: #695b78;
        font-size: .82rem;
        font-weight: 600;
    }
    [data-testid="stButton"] button:hover {
        border-color: #d9c5e8;
        color: #855da9;
    }
    .result-card {
        margin-top: 1.2rem;
        padding: 1.25rem 1.4rem;
        border: 1px solid;
        border-radius: 20px;
    }
    .result-card.spam {
        border-color: #f4c5d0;
        background: linear-gradient(120deg, #fff0f3, #fff8f8);
    }
    .result-card.ham {
        border-color: #c8e9dc;
        background: linear-gradient(120deg, #effaf5, #f8fffb);
    }
    .result-kicker {
        margin-bottom: .28rem;
        color: #81768d;
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: .1em;
    }
    .result-title {
        color: var(--ink);
        font-family: 'Trebuchet MS', Arial, sans-serif;
        font-size: 1.55rem;
        font-weight: 900;
    }
    .result-copy { margin-top: .18rem; color: #756a80; font-size: .9rem; }
    .footnote {
        padding-top: 1.2rem;
        color: #988da0;
        font-size: .78rem;
        text-align: center;
    }
    @media (max-width: 520px) {
        .block-container { padding: 2rem .85rem; }
        [data-testid="stForm"] { padding: .95rem; border-radius: 20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model, tfidf = load_artifacts()

st.markdown(
    """
    <div class="hero">
        <div class="brand-pill">💌 &nbsp; KIND INBOX</div>
        <h1>A little check for<br><span>your messages</span></h1>
        <p>Paste a text message below and let me check whether it looks like spam or a friendly ham message.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def use_example(message):
    st.session_state["message_input"] = message
    st.session_state.pop("last_result", None)


st.markdown('<div class="helper-heading">TRY A LITTLE EXAMPLE</div>', unsafe_allow_html=True)
example_columns = st.columns(2)
with example_columns[0]:
    st.button(
        "☕  Friendly message",
        on_click=use_example,
        args=("Hey, are we still meeting for coffee tomorrow?",),
        use_container_width=True,
    )
with example_columns[1]:
    st.button(
        "🎁  Suspicious message",
        on_click=use_example,
        args=("Congratulations! You have won a free prize. Call now to claim!",),
        use_container_width=True,
    )

with st.form("message_checker"):
    msg = st.text_area(
        "Your message",
        height=150,
        placeholder="Paste or type a message here...",
        key="message_input",
    )
    st.caption(f"{len(msg)} characters")
    submitted = st.form_submit_button("✨  Check this message")

if submitted:
    if not msg.strip():
        st.warning("Message ekak type karanna.")
    else:
        raw = pd.Series([msg])
        X = hstack([tfidf.transform(raw.apply(clean_text)), extra_features(raw)])
        prob = model.predict_proba(X)[0, 1]
        st.session_state["last_result"] = {
            "is_spam": prob >= THRESHOLD,
            "probability": float(prob),
        }

result = st.session_state.get("last_result")
if result:
    is_spam = result["is_spam"]
    probability = result["probability"]
    verdict = "Looks like spam" if is_spam else "Looks like ham"
    icon = "🚩" if is_spam else "🌷"
    reassurance = (
        "This message has some signs commonly found in spam."
        if is_spam
        else "This message looks more like a regular, friendly text."
    )
    card_class = "spam" if is_spam else "ham"
    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="result-kicker">YOUR MESSAGE CHECK</div>
            <div class="result-title">{icon} &nbsp; {verdict}</div>
            <div class="result-copy">{reassurance}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(probability, text=f"Spam likelihood · {probability:.1%}")
    st.caption("A helpful estimate, not a perfect guarantee.")

st.markdown(
    '<div class="footnote">Made with a little care ✿ &nbsp; Your message is checked in this session only.</div>',
    unsafe_allow_html=True,
)