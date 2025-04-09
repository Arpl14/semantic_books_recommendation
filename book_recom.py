import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.setdefaultencoding = "utf-8"
load_dotenv()

# Load dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"].fillna("cover-not-found.jpg") + "&fife=w800"

# Load documents
raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Create FAISS vectorstore
db_books = FAISS.from_documents(documents, OpenAIEmbeddings())

# Filtered vector search
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

    if author:
        filtered = filtered[filtered["authors"].str.contains(author, case=False, na=False)]

    return filtered.head(final_top_k)

def recommend_books(query, category, tone, rating, age, author):
    recommendations = retrieve_semantic_recommendations(query, category, tone, rating, age, author)
    results = []

    for _, row in recommendations.iterrows():
        desc = row["description"]
        desc_short = " ".join(desc.split()[:30]) + "..."
        authors = row["authors"].replace(";", ", ")
        caption = f"{row['title']} by {authors}: {desc_short}"
        results.append((row["large_thumbnail"], caption))

    return results

# Streamlit UI
st.set_page_config(page_title="Semantic Book Recommender", layout="wide")
st.title("📚 Semantic Book Recommendation System")

col1, col2, col3 = st.columns([2, 1, 1])
query = col1.text_input("🔎 Describe a book you’re looking for", placeholder="e.g., A story of forgiveness in a small town")
category = col2.selectbox("📂 Category", ["All"] + sorted(books["super_category"].dropna().unique()))
tone = col3.selectbox("🎭 Dominant Emotion", ["All", "joy", "sadness", "fear", "anger", "surprise", "disgust", "neutral"])

col4, col5, col6 = st.columns([1, 1, 1])
rating = col4.slider("⭐️ Minimum Rating", 0.0, 5.0, 0.0, 0.1)
age = col5.slider("📅 Maximum Book Age", 0, 100, 100)
author = col6.text_input("👩‍💼 Preferred Author", placeholder="e.g., Agatha Christie")

if st.button("🔍 Recommend"):
    results = recommend_books(query, category, tone, rating, age, author)
    st.image([img for img, _ in results], width=150)
    for _, caption in results:
        st.caption(caption)
