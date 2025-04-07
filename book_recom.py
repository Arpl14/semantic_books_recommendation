# streamlit_app.py

import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load API key
load_dotenv()

# Load books dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("cover-not-found.jpg") + "&fife=w800"

# Vector DB creation (do this only once and cache it)
@st.cache_resource
def load_vector_db():
    raw_documents = TextLoader("tagged_description.txt").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    return db

db_books = load_vector_db()

# Main search function
def retrieve_recommendations(query, category, tone, rating, age, author_keyword, initial_top_k=50, final_top_k=12):
    recs = db_books.similarity_search(query, k=initial_top_k)
    isbns = [int(doc.page_content.strip('"').split()[0]) for doc in recs]
    filtered_books = books[books["isbn13"].isin(isbns)]

    if category != "All":
        filtered_books = filtered_books[filtered_books["super_category"] == category]

    if tone != "All":
        filtered_books = filtered_books[filtered_books["dominant_emotion"] == tone.lower()]

    if rating:
        filtered_books = filtered_books[filtered_books["average_rating"] >= rating]

    if age:
        filtered_books = filtered_books[filtered_books["age_of_book"] <= age]

    if author_keyword:
        filtered_books = filtered_books[filtered_books["authors"].str.contains(author_keyword, case=False, na=False)]

    return filtered_books.head(final_top_k)

# Streamlit UI
st.set_page_config(page_title="📚 Semantic Book Recommender", layout="wide")
st.title("📚 Semantic Book Recommender")
st.markdown("_Find your next favorite read using emotions, categories, and more._")

query = st.text_input("Describe the type of book you're looking for:", "A story about hope and friendship")

col1, col2, col3 = st.columns(3)

with col1:
    category = st.selectbox("Category:", ["All"] + sorted(books["super_category"].unique()))
with col2:
    tone = st.selectbox("Dominant Emotion:", ["All", "Joy", "Sadness", "Fear", "Anger", "Surprise", "Disgust", "Neutral"])
with col3:
    author_keyword = st.text_input("Preferred author (optional):", "")

col4, col5 = st.columns(2)
with col4:
    rating = st.slider("Minimum Rating:", min_value=0.0, max_value=5.0, step=0.1, value=3.5)
with col5:
    age = st.slider("Max Age of Book (years):", min_value=0, max_value=100, step=1, value=25)

if st.button("🔍 Find Recommendations"):
    results = retrieve_recommendations(query, category, tone, rating, age, author_keyword)
    if results.empty:
        st.warning("No matching books found. Try changing your filters.")
    else:
        st.markdown("### 🔎 Top Matches")
        for _, row in results.iterrows():
            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    st.image(row["large_thumbnail"], use_column_width=True)
                with cols[1]:
                    st.subheader(f"{row['title']}")
                    st.markdown(f"**Author(s):** {row['authors']}")
                    st.markdown(f"**Category:** {row['super_category']}")
                    st.markdown(f"**Rating:** {row['average_rating']} ⭐ | **Pages:** {int(row['num_pages'])} | **Published:** ~{int(row['age_of_book'])} yrs ago")
                    st.write(" ".join(row["description"].split()[:40]) + "...")
