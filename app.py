import json
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from nlp_engine import (
    build_relation_dataframe,
    build_network_html,
    download_nltk_resources,
    get_word_relationships,
    normalize_word,
)

st.set_page_config(
    page_title="Word Relationship Explorer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

download_nltk_resources()

CUSTOM_CSS = """
<style>
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        color: white;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1300px;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(168,85,247,0.18));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 22px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .mini-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px;
        padding: 1rem;
        min-height: 110px;
    }
    .tag-pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        margin: 0.2rem;
        font-size: 0.95rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "favorites" not in st.session_state:
    st.session_state.favorites = []

with st.sidebar:
    st.title("⚙ Control Panel")
    relationship_filter = st.multiselect(
        "Relationship types",
        ["Synonyms", "Antonyms", "Hypernyms", "Hyponyms"],
        default=["Synonyms", "Antonyms", "Hypernyms", "Hyponyms"],
    )
    max_items = st.slider("Max words per section", 5, 30, 12)
    pos_filter = st.selectbox(
        "Preferred part of speech",
        ["All", "noun", "verb", "adjective", "adverb"],
        index=0,
    )
    graph_height = st.slider("Graph height", 450, 850, 620, step=10)
    st.markdown("---")
    st.caption(
        "Built with WordNet, interactive tables, and a relationship graph."
    )

st.markdown(
    """
    <div class='hero-card'>
        <h1 style='margin-bottom:0.4rem;'>🔎 Word Relationship Explorer</h1>
        <p style='font-size:1.05rem; margin-bottom:0.5rem;'>
            Explore <b>synonyms</b>, <b>antonyms</b>, <b>hypernyms</b>, and <b>hyponyms</b> of a word using NLP.
        </p>
        <p style='opacity:0.85; margin:0;'>Search a word, inspect meanings, view a live semantic network, and export results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

search_col, action_col = st.columns([4, 1])
with search_col:
    user_word = st.text_input(
        "Enter a word",
        value="bright",
        placeholder="Try words like happy, computer, book, fast, light",
    )
with action_col:
    st.write("")
    analyze = st.button("Analyze", use_container_width=True, type="primary")

quick_words = ["happy", "computer", "book", "light", "strong", "learn", "travel", "music"]
quick_cols = st.columns(len(quick_words))
for i, w in enumerate(quick_words):
    if quick_cols[i].button(w, use_container_width=True):
        user_word = w
        analyze = True

if analyze or user_word:
    clean_word = normalize_word(user_word)
    if clean_word:
        result = get_word_relationships(clean_word, max_items=max_items, pos_filter=pos_filter)

        if result["found"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_entry = {"word": clean_word, "time": timestamp}
            if history_entry not in st.session_state.history:
                st.session_state.history.insert(0, history_entry)
                st.session_state.history = st.session_state.history[:12]

            top1, top2, top3, top4 = st.columns(4)
            top1.metric("Synonyms", len(result["synonyms"]))
            top2.metric("Antonyms", len(result["antonyms"]))
            top3.metric("Hypernyms", len(result["hypernyms"]))
            top4.metric("Hyponyms", len(result["hyponyms"]))

            action1, action2, action3 = st.columns([1, 1, 5])
            with action1:
                if st.button("⭐ Save", use_container_width=True):
                    if clean_word not in st.session_state.favorites:
                        st.session_state.favorites.append(clean_word)
            with action2:
                st.download_button(
                    "⬇ JSON",
                    data=json.dumps(result, indent=2),
                    file_name=f"{clean_word}_relationships.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with action3:
                st.caption(f"Exploring the word: {clean_word}")

            tabs = st.tabs(["Overview", "Relationship Graph", "Data Table", "Word Senses", "Project Notes"])

            with tabs[0]:
                section_map = {
                    "Synonyms": result["synonyms"],
                    "Antonyms": result["antonyms"],
                    "Hypernyms": result["hypernyms"],
                    "Hyponyms": result["hyponyms"],
                }
                cols = st.columns(2)
                idx = 0
                for label, words in section_map.items():
                    if label in relationship_filter:
                        with cols[idx % 2]:
                            st.markdown(f"<div class='mini-card'><div class='section-title'>{label}</div>", unsafe_allow_html=True)
                            if words:
                                pills = "".join([f"<span class='tag-pill'>{w}</span>" for w in words])
                                st.markdown(pills, unsafe_allow_html=True)
                            else:
                                st.info(f"No {label.lower()} found for this selection.")
                            st.markdown("</div>", unsafe_allow_html=True)
                        idx += 1

                st.subheader("Definition Summary")
                if result["definitions"]:
                    for i, d in enumerate(result["definitions"], start=1):
                        st.write(f"{i}. {d}")
                else:
                    st.info("No definitions available.")

            with tabs[1]:
                graph_html = build_network_html(clean_word, result)
                components.html(graph_html, height=graph_height, scrolling=True)

            with tabs[2]:
                df = build_relation_dataframe(clean_word, result)
                if relationship_filter:
                    df = df[df["relationship"].isin([x.lower() for x in relationship_filter])]
                st.dataframe(df, use_container_width=True, hide_index=True)
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name=f"{clean_word}_relationships.csv",
                    mime="text/csv",
                )

            with tabs[3]:
                st.subheader("Word Senses from WordNet")
                if result["senses"]:
                    for i, sense in enumerate(result["senses"], start=1):
                        with st.expander(f"Sense {i}: {sense['name']}"):
                            st.write(f"**Part of speech:** {sense['pos']}")
                            st.write(f"**Definition:** {sense['definition']}")
                            if sense["examples"]:
                                st.write("**Examples:**")
                                for ex in sense["examples"]:
                                    st.write(f"- {ex}")
                            if sense["lemmas"]:
                                st.write("**Related lemmas:** " + ", ".join(sense["lemmas"]))
                else:
                    st.info("No word senses found.")

            with tabs[4]:
                st.markdown(
                    """
                    ### NLP Concepts Used
                    - **WordNet** for lexical semantic relationships.
                    - **Token normalization** to clean user input.
                    - **Semantic relationship extraction** for synonyms, antonyms, hypernyms, and hyponyms.
                    - **Interactive visualization** to make the project demo-friendly.

                    ### Why ? 
                    - Modern web interface
                    - Interactive graph
                    - Search history and favorites
                    - CSV and JSON export
                    - Expandable sense-wise analysis
                    """
                )
        else:
            st.error("No WordNet data found for this word. Try a simple English word like 'happy' or 'book'.")
    else:
        st.warning("Please enter a valid word.")

st.markdown("---")
left, right = st.columns([2, 1])

with left:
    st.subheader("Recent Searches")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("No searches yet.")

with right:
    st.subheader("Favorites")
    if st.session_state.favorites:
        for fav in st.session_state.favorites:
            st.markdown(f"- {fav}")
    else:
        st.info("No favorites saved yet.")
