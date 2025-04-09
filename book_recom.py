import os
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

os.environ["PYTHONIOENCODING"] = "utf-8"
load_dotenv()

# --- Constants ---
VECTOR_DB_PATH = "faiss_books_index"
DEFAULT_COVER = "https://via.placeholder.com/150?text=No+Cover"

# --- Load dataset ---
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna(DEFAULT_COVER) + "&fife=w800"
books["large_thumbnail"] = books["large_thumbnail"].str.replace("nan&fife=w800", DEFAULT_COVER)

# Top 20 authors by average ratings_count
top_authors = (
    books.groupby("authors")["ratings_count"]
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .index.tolist()
)

# --- Load or build FAISS vectorstore ---
if os.path.exists(VECTOR_DB_PATH):
    db_books = FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = FAISS.from_documents(documents, OpenAIEmbeddings())
    db_books.save_local(VECTOR_DB_PATH)

# --- Recommendation Logic ---
def retrieve_semantic_recommendations(query, category, tone, rating, age, author, initial_top_k=50, final_top_k=12):
    recs = db_books.similarity_search(query, k=initial_top_k)
    isbns = [int(doc.page_content.split()[0].strip('"')) for doc in recs]
    filtered = books[books["isbn13"].isin(isbns)]

    if category != "All":
        filtered = filtered[filtered["super_category"] == category]

    if tone != "All":
        filtered = filtered.sort_values(by=tone.lower(), ascending=False)

    if rating:
        filtered = filtered[filtered["average_rating"] >= rating]

    if age:
        filtered = filtered[filtered["age_of_book"] <= age]

    if author and author != "No preference":
        filtered = filtered[filtered["authors"].str.contains(author, case=False, na=False)]

    return filtered.head(final_top_k)

# --- Streamlit UI ---
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")
st.title("📚 Semantic Book Recommendation System")

col1, col2, col3 = st.columns([2, 1, 1])
query = col1.text_input("🔎 Describe a book you’re looking for", placeholder="e.g., A story of forgiveness in a small town")
category = col2.selectbox("📂 Category", ["All"] + sorted(books["super_category"].dropna().unique()))
tone = col3.selectbox("🎭 Dominant Emotion", ["All", "joy", "sadness", "fear", "anger", "surprise", "disgust", "neutral"])

col4, col5, col6 = st.columns([1, 1, 1])
rating_display = col4.selectbox("⭐️ Minimum Rating", ["No preference", 1, 2, 3, 4, 5])
rating = 0 if rating_display == "No preference" else float(rating_display)
age = col5.slider("📅 Show books up to how old? (in years)", 0, 100, 100)

# Preferred author – combo of dropdown + type
author = col6.selectbox(
    "👩‍💼 Preferred Author (or type your own)",
    options=["No preference"] + top_authors,
    index=0,
    placeholder="Choose a top-selling author or type your own",
)
# --- Results ---
if st.button("🔍 Recommend"):
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
                    st.write(row["description"])
