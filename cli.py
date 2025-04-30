import pandas as pd
import numpy as np  
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

torch.classes.__path__ = []

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

index = faiss.read_index("./data/movie_index.faiss")
movie_metadata = np.load('./data/movie_metadata.npy', allow_pickle=True)

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
rag_pipeline = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)

user_input = input("Enter your question: ")
print(f"\n You asked: {user_input}")

query_embedding = embed_model.encode(user_input, convert_to_numpy=True)
D, I = index.search(np.array([query_embedding], dtype=np.float32), k=5)
retrieved_movies = [movie_metadata[i] for i in I[0]]

context = ""
for movie in retrieved_movies:
    # print(f"\n   Movie: {movie['title']}")
    # print(f"   Tagline: {movie['tagline']}")
    # print(f"   Budget: {movie['budget']}")
    # print(f"   Revenue: {movie['revenue']}")
    # print(f"   Runtime: {movie['runtime']}")
    # print(f"   Genres: {movie['genres']}")
    context += f"Title: {movie['title']}\nGenres: {movie['genres']}\nBudget: {movie['budget']}\nRevenue: {movie['revenue']}\n\n"

prompt = f"""You are a movie critic. Based on the following movie data, recommend a good movie to the user: "{user_input}"

{context}
Your recommendation:"""

response = rag_pipeline(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]['generated_text']
print("\n AI Recommendation:\n")
print(response.split("Your recommendation:")[-1].strip())