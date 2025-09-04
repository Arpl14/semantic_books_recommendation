import os
import re
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from difflib import get_close_matches

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# --- Setup ---
os.environ["PYTHONIOENCODING"] = "utf-8"
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY not found in environment. Embeddings may fail.")

VECTOR_DB_PATH = "faiss_books_index"
DEFAULT_COVER = "https://via.placeholder.com/150?text=No+Cover"
TAGGED_PATH = "tagged_description.txt"

# --- Load dataset ---
@st.cache_data(show_spinner=False)
def load_books():
    df = pd.read_csv("books_with_emotions.csv")
    # robust thumbnail handling
    thumbs = df["thumbnail"].fillna("")
    # only append fife param if URL exists
    thumbs = np.where(thumbs.str.len() > 0, thumbs + ("&fife=w800" if "~" not in thumbs else ""), DEFAULT_COVER)
    df["large_thumbnail"] = pd.Series(thumbs).replace({"nan&fife=w800": DEFAULT_COVER})
    df["authors_clean"] = df["authors"].fillna("").str.lower()
    return df

books = load_books()

# --- Build or Load FAISS Vectorstore ---
@st.cache_resource(show_spinner=True)
def get_vectorstore():
    embeddings = OpenAIEmbeddings()  # relies on env var
    if os.path.exists(VECTOR_DB_PATH):
        try:
            return FAISS.load_local(
                VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.info("Rebuilding FAISS index because loading failed.")
    # Build from source
    if not os.path.exists(TAGGED_PATH):
        st.error(f"Missing '{TAGGED_PATH}'. Place it next to this script.")
        st.stop()

    raw_documents = TextLoader(TAGGED_PATH, encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    documents = splitter.split_documents(raw_documents)
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(VECTOR_DB_PATH)
    return db

db_books = get_vectorstore()

# --- Helpers ---
ISBN_RE = re.compile(r'^\s*"?(\d{10,13})"?')

def _safe_isbn_from_page(page_text: str):
    """
    Expects each chunk in tagged_description to start with ISBN,
    e.g., 9780671027032 "Description text ..."
    """
    m = ISBN_RE.match(page_text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

# --- Recommendation Logic ---
def retrieve_semantic_recommendations(
    query, category, tone, rating, age, author,
    initial_top_k=200, final_top_k=20
):
    filtered = books.copy()

    # Author fuzzy filter
    if author:
        author_clean = author.lower().strip()
        all_authors = books["authors_clean"].dropna().unique().tolist()
        match = get_close_matches(author_clean, all_authors, n=1, cutoff=0.6)
        if match:
            filtered = filtered[filtered["authors_clean"].str.contains(re.escape(match[0]), na=False)]

    # Category
    if category != "All":
        filtered = filtered[filtered["super_category"] == category]

    # Emotion sorting (assumes emotion columns exist and are numeric)
    if tone != "All" and tone.lower() in filtered.columns:
        filtered = filtered.sort_values(by=tone.lower(), ascending=False)

    # Rating
    if rating and float(rating) > 0:
        filtered = filtered[filtered["average_rating"] >= float(rating)]

    # Age
    if age is not None:
        filtered = filtered[filtered["age_of_book"] <= int(age)]

    # Semantic retrieval (restrict to filtered subset)
    if query:
        recs = db_books.similarity_search(query, k=initial_top_k)
        isbns = []
        for doc in recs:
            isbn = _safe_isbn_from_page(doc.page_content)
            if isbn is not None:
                isbns.append(isbn)
        if len(isbns) == 0:
            return filtered.head(final_top_k)
        filtered = filtered[filtered["isbn13"].isin(isbns)]

        # Optional: preserve semantic order
        order = {isbn: i for i, isbn in enumerate(isbns)}
        filtered = filtered.assign(_ord=filtered["isbn13"].map(order)).sort_values("_ord").drop(columns="_ord", errors="ignore")

    return filtered.head(final_top_k)

# --- UI ---
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")
st.title("\U0001F4DA Semantic Book Recommendation System")

st.markdown(
    """
    <div style='display: flex; justify-content: space-between; align-items: center; margin-top:-10px; margin-bottom:-10px;'>
        <span style='font-size: 18px; color: gray;'>Find the perfect book to read — for the love of stories, discovery, and imagination.</span>
        <span style='font-size: 18px; color: gray; '>- Arpita Lonakadi</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Row 1: Query and Author
col1, col2 = st.columns([2, 1])
query = col1.text_input(" Describe a book you’re looking for", placeholder="e.g., A story of forgiveness in a small town")
author = col2.text_input("Preferred Author", placeholder="e.g., Paulo Coelho")

# Row 2: Rating, Emotion, Category, Age
col3, col4, col5, col6 = st.columns([1, 1, 1, 2])
rating_display = col3.selectbox("Minimum Rating", ["No preference", 1, 2, 3, 4, 5])
rating = 0 if rating_display == "No preference" else float(rating_display)
tone = col4.selectbox(" Dominant Emotion", ["All", "joy", "sadness", "fear", "anger", "surprise", "disgust", "neutral"])
category = col5.selectbox("Category", ["All"] + sorted(books["super_category"].dropna().unique()))
age = col6.slider(" Age of book (in years)", 0, 100, 100)

# --- Results ---
if st.button("\U0001F50D Recommend"):
    try:
        recommendations = retrieve_semantic_recommendations(query, category, tone, rating, age, author)
    except Exception as e:
        st.error(f"Recommendation error: {e}")
        st.stop()

    if recommendations.empty:
        st.warning("No recommendations found. Try adjusting your filters.")
    else:
        cols = st.columns(3)
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            col = cols[idx % 3]
            with col:
                st.image(row.get("large_thumbnail", DEFAULT_COVER), width=150)
                title = row.get("title", "Untitled")
                authors = row.get("authors", "Unknown author")
                st.markdown(f"**{title}** by *{authors}*")
                desc = str(row.get("description", "") or "")
                short_desc = " ".join(desc.split()[:20]) + ("..." if len(desc.split()) > 20 else "")
                with st.expander(short_desc if short_desc else "Description"):
                    st.markdown(f"**Full Description:**\n\n{desc}")
