"""
app.py
------
Food Recommender — Streamlit UI with three tabs:
  Tab 1: Semantic Search
  Tab 2: Calorie-Range Filtered Search
  Tab 3: RAG Chatbot (ChromaDB retrieval + Groq streaming)
"""

import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from ingest import get_collection, ingest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_MODEL    = "llama-3.3-70b-versatile"
TOP_K_CHAT    = 5          # docs retrieved per chatbot message
SYSTEM_PROMPT = (
    "You are a friendly food recommendation assistant. Your job is "
    "to always recommend something useful to the user, no matter what they ask. "
    "Use ONLY the dishes provided in the context below.\n\n"
    "Rules:\n"
    "- Always recommend at least 2 dishes, even if the match is not perfect\n"
    "- If the user's exact request is not available, recommend the closest available "
    "option and honestly explain why it is a good alternative\n"
    "- Never say you cannot help or that nothing matches\n"
    "- Never invent dishes that are not in the context\n"
    "- Always mention the dish name, calories, and one specific reason "
    "why it suits the user's request\n"
    "- Be warm, direct, and confident — never apologetic or overly formal"
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Foodie — Food Recommender",
    page_icon="🍽️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS — warm card styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .food-card {
        background: #2e1a08;
        border: 1px solid #c87941;
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 20px;
    }
    .food-card h3 {
        margin: 0 0 12px 0;
        color: #f0c070;
        font-size: 1.25rem;
        line-height: 1.3;
    }
    .food-card .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 14px;
    }
    .food-card .badge {
        display: inline-block;
        background: #c87941;
        color: #1a1008;
        border-radius: 6px;
        padding: 3px 11px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    /* sim score colour variants */
    .sim-score        { float: right; font-size: 1.05rem; font-weight: 700; padding: 2px 10px; border-radius: 20px; }
    .sim-score.green  { background: #1a3d1a; color: #5de05d; }
    .sim-score.amber  { background: #3d2e00; color: #f0b429; }
    .sim-score.red    { background: #3d0e0e; color: #f06060; }
    .card-divider {
        border: none;
        border-top: 1px solid #3d2510;
        margin: 0 0 14px 0;
    }
    .food-card .taste-line {
        font-size: 0.85rem;
        color: #c8a060;
        margin: 0 0 10px 0;
        font-style: italic;
    }
    .food-card .desc {
        color: #f5e6c8;
        line-height: 1.65;
        font-size: 0.92rem;
        margin: 0;
    }
    /* Streamlit Tabs Customization — Warm Espresso Theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        justify-content: center;
        border-bottom: 1px solid #3d2510;
        padding-bottom: 4px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #c8a882;
        font-weight: 600;
        font-size: 1rem;
        padding: 8px 16px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f0c070;
    }
    .stTabs [aria-selected="true"] {
        color: #c87941 !important;
        background-color: rgba(200, 121, 65, 0.12) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #c87941 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources — created once per session, shared across reruns
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading food database...")
def load_collection():
    col = get_collection()
    ingest(col)
    return col


@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    return Groq(api_key=api_key)


collection  = load_collection()
groq_client = load_groq_client()


# ---------------------------------------------------------------------------
# Helper — render one food result card
# ---------------------------------------------------------------------------
def render_card(meta: dict, doc: str, distance: float | None = None) -> None:
    sim_html = ""
    if distance is not None:
        sim = (1.0 - distance) * 100
        if sim >= 70:
            css_class = "green"
            indicator  = "&#9679;"   # filled circle
        elif sim >= 50:
            css_class = "amber"
            indicator  = "&#9679;"
        else:
            css_class = "red"
            indicator  = "&#9679;"
        sim_html = (
            f'<span class="sim-score {css_class}">'
            f'{indicator} {sim:.1f}% match</span>'
        )

    st.markdown(
        f"""
        <div class="food-card">
            <h3>{meta["name"]} {sim_html}</h3>
            <div class="badge-row">
                <span class="badge">{meta["cuisine"]}</span>
                <span class="badge">{meta["dietary_type"]}</span>
                <span class="badge">{meta["cooking_method"]}</span>
                <span class="badge">{meta["calories"]} kcal</span>
            </div>
            <hr class="card-divider">
            <p class="taste-line">Taste: {meta["taste_profile"]}</p>
            <p class="desc">{meta["description"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# App header — branded Foodie welcome block
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0 0.5rem 0;">
        <div style="
            font-family: 'Cormorant Garamond', Georgia, 'Times New Roman', serif;
            font-size: 3rem;
            font-style: italic;
            font-weight: 600;
            color: #c87941;
            letter-spacing: 0.02em;
            line-height: 1.1;
            margin-bottom: 0.6rem;
        ">
            🍽️ &nbsp;<em>Foodie</em>
        </div>
        <div style="
            font-size: 1rem;
            color: #c8a882;
            font-style: italic;
            margin-bottom: 1.4rem;
        ">
            Welcome to <em>Foodie</em> &mdash; discover dishes by craving, calorie range, or conversation.
        </div>
        <div style="
            height: 1px;
            background: linear-gradient(to right, transparent, #c87941, transparent);
            margin: 0 auto;
            max-width: 600px;
        "></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(
    ["🔍 Semantic Search", "📊 Calorie Filter", "💬 AI Chat"]
)

# ===========================================================================
# TAB 1 — Semantic Search
# ===========================================================================
with tab1:
    st.markdown("<h2 style='text-align: center; color: #f0c070; font-size: 1.5rem; margin-top: 0.5rem;'>What are you craving?</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #c8a882; margin-bottom: 1.5rem;'>Describe what you feel like eating in plain English.</p>", unsafe_allow_html=True)

    query1    = st.text_input(
        "Your craving",
        placeholder='e.g. "something spicy and grilled" or "light vegetarian lunch"',
        key="tab1_query",
    )
    n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if st.button("Search", key="tab1_search"):
        if not query1.strip():
            st.warning("Please type something you're craving.")
        else:
            with st.spinner("Searching..."):
                results = collection.query(
                    query_texts=[query1],
                    n_results=n_results,
                    include=["documents", "distances", "metadatas"],
                )

            ids       = results["ids"][0]
            distances = results["distances"][0]
            metas     = results["metadatas"][0]
            docs      = results["documents"][0]

            if not ids:
                st.info("No results found.")
            else:
                st.markdown(f"**{len(ids)} results for:** _{query1}_")
                for meta, doc, dist in zip(metas, docs, distances):
                    render_card(meta, doc, distance=dist)

# ===========================================================================
# TAB 2 — Calorie-Range Filtered Search
# ===========================================================================
with tab2:
    st.markdown("<h2 style='text-align: center; color: #f0c070; font-size: 1.5rem; margin-top: 0.5rem;'>Search by Calorie Range</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #c8a882; margin-bottom: 1.5rem;'>Filter is applied inside ChromaDB using <code>$gte</code>/<code>$lte</code> on metadata.</p>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        min_cal = st.number_input("Min Calories", min_value=0, max_value=2000, value=0, step=50)
    with col_b:
        max_cal = st.number_input("Max Calories", min_value=0, max_value=2000, value=900, step=50)

    query2 = st.text_input(
        "Optional semantic query within this range",
        placeholder='e.g. "something light and fresh" (leave blank to see all)',
        key="tab2_query",
    )
    n_results2 = st.slider("Number of results", min_value=1, max_value=10, value=5, key="tab2_slider")

    # Build ChromaDB where filter — applied INSIDE the HNSW query
    cal_filter = {
        "$and": [
            {"calories": {"$gte": int(min_cal)}},
            {"calories": {"$lte": int(max_cal)}},
        ]
    }

    if st.button("Search", key="tab2_search"):
        if min_cal > max_cal:
            st.error("Min calories must be less than or equal to max calories.")
        else:
            with st.spinner("Filtering..."):

                if query2.strip():
                    # Path A: semantic query + calorie filter pushed into ChromaDB
                    results = collection.query(
                        query_texts=[query2],
                        n_results=n_results2,
                        where=cal_filter,
                        include=["documents", "distances", "metadatas"],
                    )
                    ids       = results["ids"][0]
                    distances = results["distances"][0]
                    metas     = results["metadatas"][0]
                    docs      = results["documents"][0]

                    if not ids:
                        st.info(f"No results between {min_cal} and {max_cal} kcal.")
                    else:
                        st.markdown(
                            f"**{len(ids)} results** for _{query2}_ "
                            f"between **{min_cal}–{max_cal} kcal**"
                        )
                        for meta, doc, dist in zip(metas, docs, distances):
                            render_card(meta, doc, distance=dist)

                else:
                    # Path B: no query — browse all docs in calorie range (no ranking)
                    results = collection.get(
                        where=cal_filter,
                        include=["documents", "metadatas"],
                    )
                    ids   = results["ids"]
                    metas = results["metadatas"]
                    docs  = results["documents"]

                    if not ids:
                        st.info(f"No dishes found between {min_cal} and {max_cal} kcal.")
                    else:
                        st.markdown(
                            f"**{len(ids)} dishes** between "
                            f"**{min_cal}–{max_cal} kcal** (unranked)"
                        )
                        for meta, doc in zip(metas, docs):
                            render_card(meta, doc, distance=None)

# ===========================================================================
# TAB 3 — RAG Chatbot
# ===========================================================================
with tab3:
    st.markdown("<h2 style='text-align: center; color: #f0c070; font-size: 1.5rem; margin-top: 0.5rem;'>Chat with Foodie AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #c8a882; margin-bottom: 1.5rem;'>Ask anything about food — our AI retrieves relevant dishes from our database before answering.</p>", unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ── Clear chat button ──────────────────────────────────────────────────
    if st.button("Clear Chat", key="clear_chat"):
        st.session_state["messages"] = []
        st.rerun()

    # ── Groq API key guard ─────────────────────────────────────────────────
    if groq_client is None:
        st.error(
            "GROQ_API_KEY not found. Create a `.env` file in the "
            "`food_recommender/` folder with `GROQ_API_KEY=your_key_here`."
        )
        st.stop()

    # ── Render conversation history ────────────────────────────────────────
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ─────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about food, cravings, or dietary needs...")

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Step 1 — Retrieve top-K relevant food items from ChromaDB (include distances)
        retrieval = collection.query(
            query_texts=[user_input],
            n_results=TOP_K_CHAT,
            include=["documents", "metadatas", "distances"],
        )
        retrieved_metas     = retrieval["metadatas"][0]
        retrieved_docs      = retrieval["documents"][0]
        retrieved_distances = retrieval["distances"][0]

        # Step 2 — Format context block for the prompt
        # If ALL retrieved results are low-quality (distance > 0.6 means < 40% match),
        # prepend a note so Groq frames its answer as alternatives, not perfect matches.
        all_low_quality = all(d > 0.6 for d in retrieved_distances)

        context_lines = []
        for i, (meta, doc) in enumerate(zip(retrieved_metas, retrieved_docs), start=1):
            context_lines.append(
                f"{i}. {meta['name']} ({meta['cuisine']}, {meta['dietary_type']}, "
                f"{meta['calories']} kcal, {meta['cooking_method']})\n   {meta['description']}"
            )
        context_block = "\n\n".join(context_lines)

        if all_low_quality:
            context_block = (
                "Note: these are the closest available matches, not exact matches. "
                "Frame your response as alternatives and explain why each is a reasonable "
                "substitute for what the user asked for.\n\n"
            ) + context_block

        # Step 3 — Build Groq messages list
        groq_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "FOOD CONTEXT (use only these dishes in your recommendations):\n\n"
                    + context_block
                ),
            },
        ]
        # Append full conversation history for multi-turn coherence
        groq_messages.extend(st.session_state["messages"])
        # Append current user turn
        groq_messages.append({"role": "user", "content": user_input})

        # Step 4 — Stream Groq response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                stream = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=groq_messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=1024,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response += delta
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"Error calling Groq API: {e}"
                response_placeholder.error(full_response)

        # Step 5 — Append both turns to session state for next round
        st.session_state["messages"].append({"role": "user",      "content": user_input})
        st.session_state["messages"].append({"role": "assistant",  "content": full_response})
