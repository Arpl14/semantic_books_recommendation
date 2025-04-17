# Semantic Book Recommendation System
https://semanticbooksrecommendation-tm7pdn3kxsrmmttvs2jjsc.streamlit.app/
![books_search](https://github.com/user-attachments/assets/c8575e3d-ccd0-4dea-87d0-0fedf637d32d)




This is a full-stack LLM-powered book recommendation system that semantically understands user queries and suggests relevant books based on a combination of natural language input, filters, and emotion-aware descriptions. Built using Streamlit, LangChain, Hugging Face Transformers, and OpenAI embeddings, it demonstrates advanced capabilities in vector search, NLP pipelines, and human-centered AI.
---

## Project Overview

This project leverages large language models (LLMs) to enhance traditional content-based recommendations by integrating semantic search, zero-shot classification, and emotion-aware filtering on a curated dataset of 7,000+ books.

---

## Key Features

- **Semantic Search**: Contextual book matching using OpenAI embeddings and FAISS vector store.
- **Emotion Detection**: Description-level emotion scoring using `distilroberta-base` from Hugging Face.
- **Zero-Shot Classification**: Automatic super-category prediction for unstructured genres using `facebook/bart-large-mnli`.
- **Author Name Matching**: Fuzzy matching to capture spelling variations (e.g., “J.K. Rowling” vs “Rowling, J.K.”).
- **Dynamic Filtering**: Filter by rating, age, emotion, author, and category even without a query.

---

## Technical Stack

- **Languages & Libraries**: Python, Pandas, NumPy, Streamlit, Matplotlib, Seaborn
- **LLM/AI Tools**:
  - `OpenAI Embeddings` for semantic vectorization
  - `LangChain` for FAISS-based vector DB integration
  - `Hugging Face Transformers` for:
    - Zero-shot classification (`bart-large-mnli`)
    - Emotion analysis (`distilroberta-base`)
- **NLP/ML Techniques**:
  - Preprocessing & tokenization
  - Sentence-level emotion aggregation
  - Embedding-based similarity retrieval
  - Category bucketing and imputation via transformers

---

## Implementation Highlights

- Cleaned and preprocessed raw metadata (title, author, ratings, categories, descriptions).
- Vectorized descriptions using `tagged_description = ISBN + description` for accurate indexing.
- Built a FAISS vector store on top of OpenAI-embedded text chunks.
- Used `LangChain`’s `TextLoader` + `CharacterTextSplitter` to support scalable semantic chunking.
- Imputed missing categories via zero-shot classification on descriptions.
- Classified dominant emotions per book using Hugging Face's emotion classifier.

---

## How It Works

1. **Preprocessing**: Cleans metadata, handles missing values, engineers features.
2. **Categorization**: Super-categories assigned using rule-based + LLM inference.
3. **Emotion Detection**: Runs each description through a sentence-wise emotion pipeline.
4. **Vector Store**: Book descriptions are indexed as vector embeddings.
5. **Recommendation**:
   - Applies user filters (author, category, emotion, rating).
   - Runs semantic search (if query is given) on prefiltered subset.
   - Displays top N relevant books with expandable summaries.

---

## Achievements

- Enabled **semantic recommendation** without needing structured labels or keywords.
- Boosted discoverability using **multi-modal filters** and **LLM-classified emotions**.
- Solved fuzzy search challenges with **intelligent author name correction**.
- Addressed category gaps with **zero-shot generalization** from unstructured text.
- Delivered a fully interactive, front-end app using **Streamlit**.

---

## How to Use
Use the app made by me and find out the favourite books you want to read. 
https://semanticbooksrecommendation-tm7pdn3kxsrmmttvs2jjsc.streamlit.app/

To clone : 
1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
2. Add your own Open AI API key in .env file : `OPENAI_API_KEY=your_key_here`
3. Run the app : `streamlit run book_recom.py`


## Conclusion 
This project blends NLP, LLMs, MLOPs and information retrieval to build a powerful, human-centric book recommendation system. It’s a proof-of-concept for how embeddings and transformer-based reasoning can unlock richer, more flexible user experiences in discovery platforms.
