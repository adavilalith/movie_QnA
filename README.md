# 🎬 Movie Q&A with RAG (Retrieval-Augmented Generation)

This project is a simple, fully local **RAG (Retrieval-Augmented Generation)** system that allows users to ask natural language questions about movies and get intelligent, context-aware answers — without needing an internet connection or paid APIs.

It combines:  
- 🧠 Semantic search using SentenceTransformers  
- 🔍 Fast vector similarity with FAISS  
- 🤖 Local LLM generation using Mistral  

---

## 🚀 Features

- Ask questions like _"interesting sci-fi movies"_ or _"top romance recommendations"_  
- Retrieves the top matching movie chunks based on overviews and metadata  
- Generates a smart, human-like response using a local LLM  
- Runs fully offline, free, and fast  

---

## 📁 Dataset

The dataset is based on the Kaggle notebook:  
Movie QnA with RAG (by Adavila): https://www.kaggle.com/code/adavilalith/movie-qna-rag

You must extract metadata such as:  
- title, tagline, budget, revenue, runtime, genres, etc.  
- Store this in a NumPy `.npy` file or `.json` to pair with the FAISS index  

---

## 📦 Dependencies

Install the required Python libraries:

- sentence-transformers  
- faiss-cpu  
- transformers  
- accelerate  
- torch  
- numpy  
- pandas  

For GPU support, use the CUDA version of torch from the official PyTorch index.

---

## 🛠 How It Works

1. **Embedding & Indexing**:  
   Movie overviews and metadata are embedded using `all-MiniLM-L6-v2` and stored in a FAISS index with metadata saved separately.

2. **Retrieval**:  
   A user query is embedded and searched against the FAISS index. The top `k` similar movie entries are retrieved.

3. **Generation**:  
   The retrieved data is used as context for a local LLM (e.g., Mistral), which generates a final answer.

---

## 💻 Example Usage

User Input: _"interesting scifi movies"_  
Returned Movie: _Interstellar_  
- Tagline: Mankind's next step will be our greatest.  
- Budget: 165000000  
- Revenue: 677000000  
- Runtime: 169  
- Genres: Sci-Fi, Adventure  

AI Response:  
_If you're into mind-bending sci-fi with emotional depth, check out "Interstellar". Its massive scale, scientific grounding, and visual storytelling make it a standout._

---

## 🧠 Models Used

- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`  
- LLM (offline): `mistralai/Mistral-7B-Instruct-v0.1`  
  You can also swap in TinyLlama or other smaller open-source models for low-resource systems.

---

## 📂 File Structure

movie-rag/  
├── data/  
│   ├── movie_index.faiss         (FAISS vector index)  
│   └── movie_metadata.npy        (Movie info like title, genres, etc.)  
├── main.py                       (Full RAG pipeline)  
├── README.md  

---

## 🧩 TODO / Possible Improvements

- Add UI using Gradio or Streamlit  
- Allow filtering by genre or year  
- Improve generation quality with prompt tuning  

---
