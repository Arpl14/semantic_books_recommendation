import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from difflib import get_close_matches

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# --- Setup ---
os.environ["PYTHONIOENCODING"] = "utf-8"
load_dotenv()

VECTOR_DB_PATH = "faiss_books_index"
DEFAULT_COVER = "https://via.placeholder.com/150?text=No+Cover"

# --- Load dataset ---
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna(DEFAULT_COVER) + "&fife=w800"
books["large_thumbnail"] = books["large_thumbnail"].str.replace("nan&fife=w800", DEFAULT_COVER)
books["authors_clean"] = books["authors"].fillna("").str.lower()

# --- Load or Build FAISS Vectorstore ---
if os.path.exists(VECTOR_DB_PATH):
    db_books = FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = FAISS.from_documents(documents, OpenAIEmbeddings())
    db_books.save_local(VECTOR_DB_PATH)

# --- Recommendation Logic ---
def retrieve_semantic_recommendations(query, category, tone, rating, age, author, initial_top_k=200, final_top_k=20):
    # Step 1: Start with the full dataset
    filtered = books.copy()

    # Step 2: Apply filters first
    if author:
        author_clean = author.lower().strip()
        all_authors = books["authors_clean"].dropna().unique()
        match = get_close_matches(author_clean, all_authors, n=1, cutoff=0.6)
        if match:
            filtered = filtered[filtered["authors_clean"].str.contains(match[0], na=False)]

    if category != "All":
        filtered = filtered[filtered["super_category"] == category]

    if tone != "All":
        filtered = filtered.sort_values(by=tone.lower(), ascending=False)

    if rating:
        filtered = filtered[filtered["average_rating"] >= rating]

    if age:
        filtered = filtered[filtered["age_of_book"] <= age]

    # Step 3: If query is provided, apply similarity search on filtered subset
    if query:
        recs = db_books.similarity_search(query, k=initial_top_k)
        isbns = [int(doc.page_content.split()[0].strip('"')) for doc in recs]
        filtered = filtered[filtered["isbn13"].isin(isbns)]

    return filtered.head(final_top_k)

# --- UI ---
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")

st.title("\U0001F4DA Semantic Book Recommendation System")

st.markdown(
    """
    <div style='font-size:18px; color:gray; margin-top:-10px;'>
        Find the perfect book to read — for the love of stories, discovery, and imagination.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='text-align: right; font-weight: bold; margin-top:10px;'>- Arpita Lonakadi</div>",
    unsafe_allow_html=True,
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
    recommendations = retrieve_semantic_recommendations(query, category, tone, rating, age, author)

    if recommendations.empty:
        st.warning("No recommendations found. Try adjusting your filters.")
    else:
        cols = st.columns(3)
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            col = cols[idx % 3]
            with col:
                st.image(row["large_thumbnail"], width=150)
                st.markdown(f"**{row['title']}** by *{row['authors']}*")
                short_desc = " ".join(row["description"].split()[:20]) + "..."
                with st.expander(short_desc):
                    st.markdown(f"**Full Description:**\n\n{row['description']}")
