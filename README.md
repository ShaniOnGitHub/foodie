# 🍽️ *Foodie* — AI-Powered Food Recommendation System

> A semantic food discovery app built with ChromaDB, Groq, and Streamlit.  
> Search by craving, filter by calories, or chat with an AI that only recommends real dishes.

---

## What is Foodie?

**Foodie** is a production-quality RAG (Retrieval-Augmented Generation) application that lets you discover food through natural language. Instead of keyword search or category dropdowns, you describe what you feel like eating — and the system finds the closest matching dishes from a curated database of 40 international recipes.

Every recommendation is grounded: the AI never invents dishes. It only recommends what exists in the vector store.

---

## Features

### Tab 1 — Semantic Search
Type a free-text craving like `"something spicy and grilled"` or `"light vegan lunch under 400 calories"`. ChromaDB embeds your query and retrieves the most semantically similar dishes, ranked by cosine similarity. Each result card shows:
- Dish name with a **colour-coded similarity score** (green / amber / red)
- Cuisine, dietary type, cooking method, calorie count
- Taste profile and full description

### Tab 2 — Calorie Range Filter
Set a min/max calorie range. The filter is applied **inside ChromaDB** using `$gte`/`$lte` operators on metadata — no Python-side post-filtering. Optionally add a semantic query to rank results within that range.

### Tab 3 — RAG Chatbot
A multi-turn chat interface powered by **Groq's `llama-3.3-70b-versatile`** model. For every message:
1. ChromaDB retrieves the top 5 most relevant dishes (by cosine similarity)
2. The retrieved dishes are passed as grounded context to the LLM
3. Groq streams a response that only references real dishes from the database
4. If the match quality is low (all distances > 0.6), the context is annotated so the AI frames its answer as "closest alternatives" rather than exact matches

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| **Vector store** | ChromaDB PersistentClient | Disk-backed, no server needed |
| **Embedding model** | `mixedbread-ai/mxbai-embed-large-v1` (1024-dim) | High retrieval quality for food descriptions |
| **LLM** | Groq `llama-3.3-70b-versatile` | Fast inference, streaming support |
| **UI** | Streamlit | Rapid prototyping, native chat primitives |
| **No LangChain / LlamaIndex** | Raw API calls only | Explicit, understandable data flow |

---

## Dataset

40 food items covering:
- **Cuisines**: Italian, Japanese, Indian, Mexican, American, Middle Eastern, Thai, Chinese, French, Spanish, Belgian
- **Dietary types**: vegan, vegetarian, non-vegetarian, gluten-free
- **Calorie range**: 45 kcal (Miso Soup) → 820 kcal (BBQ Burger)
- **Cooking methods**: grilled, baked, fried, raw, steamed, slow-cooked
- **Categories**: savory mains, street food, salads, soups, desserts

Each item has a rich 2–3 sentence natural language description that is embedded as a single vector — no chunking needed for structured data.

---

## Project Structure

```
foodie/
├── app.py                  # Streamlit UI — 3 tabs + branded header
├── ingest.py               # ChromaDB setup + upsert pipeline
├── food_data.json          # 40 curated food items
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
└── .streamlit/
    └── config.toml         # Warm amber/brown theme
```

The `chroma_data/` directory is auto-created on first run and gitignored.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ShaniOnGitHub/foodie.git
cd foodie
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key

```bash
cp .env.example .env
# Edit .env and add your key:
# GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 4. Run the app

```bash
streamlit run app.py
```

**First startup** will download the `mxbai-embed-large-v1` embedding model (~1.3GB) and embed all 40 dishes. This is a one-time cost — every subsequent run starts instantly from the persisted ChromaDB store.

---

## How the RAG Pipeline Works

```
User types a message
        ↓
ChromaDB embeds query with mxbai-embed-large-v1
        ↓
HNSW cosine nearest-neighbour search → top 5 dishes
        ↓
Distances checked: if all > 0.6, annotate as "closest alternatives"
        ↓
Context block built from retrieved dish metadata + descriptions
        ↓
Groq API called with: system prompt + context + conversation history + user message
        ↓
Response streamed token-by-token into st.chat_message("assistant")
        ↓
Both turns appended to st.session_state for multi-turn memory
```

---

## Key Design Decisions

- **`upsert()` not `add()`** — ingestion is always idempotent; re-running never crashes
- **Calories stored as `int` in metadata** — required for ChromaDB's `$gte`/`$lte` numeric operators
- **One vector per food item** — no chunking; each dish is a single embeddable unit
- **EF bound at collection creation** — embedding function is set once; all operations inherit it automatically
- **`@st.cache_resource`** — ChromaDB client and Groq client created once per session, not on every rerun

---

## Requirements

```
streamlit>=1.35.0
chromadb>=0.5.0
groq>=0.9.0
python-dotenv>=1.0.0
sentence-transformers>=3.0.0
```

---

## License

MIT
