import streamlit as st
import pandas as pd
import numpy as np  
from sentence_transformers import SentenceTransformer
import faiss
import torch
torch.classes.__path__ = []

model = SentenceTransformer('all-miniLM-L6-v2')
st.title("Movie QnA App")
index = faiss.read_index("./data/movie_index.faiss")
movie_metadata = np.load('./data/movie_metadata.npy', allow_pickle=True)

user_input = st.text_input("Enter your question:")

if user_input:
    st.write(f"You asked: {user_input}")
    query_embedding = model.encode(user_input,convert_to_numpy=True)
    I = index.search(np.array([query_embedding], dtype=np.float32), k=3)
    retrieved_movies = [movie_metadata[i] for i in I[0]]
    for movie in retrieved_movies:
        st.write(f"Movie: {movie['movie_name']}")
        st.write(f"Director: {movie['director']}")
        st.write(f"Language: {movie['language']}")
        st.write(f"Rating: {movie['avg_rating']}")
        st.write(f"Votes: {movie['avg_votes']}")
        st.write(f"Budget: {movie['budget']}")
        st.write(f"Genres: {movie['genres']}")
        st.write("------------")