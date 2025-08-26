🏥 Medical Chat Assistant (Custom GPT + RAG)

A domain-specific medical chat assistant built with a custom GPT-style LLM and Retrieval-Augmented Generation (RAG).
The assistant is designed to handle medical queries by combining generative reasoning with factual retrieval from a medical knowledge base.

✨ Features

🧠 Custom GPT-style Transformer trained with attention mechanism.

🔡 Custom Tokenizer for encoding/decoding medical text.

📚 RAG Pipeline using FAISS for document retrieval.

⚡ FastAPI Backend exposing inference and retrieval endpoints.

🎨 Streamlit Frontend for user-friendly chat interface.

🐳 Dockerized Deployment (FastAPI + Streamlit with Docker Compose).


📂 Project Structure
LLM-Project/
├── src/
│   ├── api/                 # FastAPI backend
│   ├── RAG/                 # RAG pipeline (retrieval + generation)
│   ├── model/               # GPT-style transformer model
│   ├── tokenizer/           # Custom tokenizer logic
│   └── inference_repo/      # Model loading, checkpoints, inference
│   ├── app/                 # streamlit app
│   └── data/                # Medical document used for RAG
│   
|    
│
├── docker/
│   ├── fastapi.Dockerfile   # Backend container
│   └── streamlit.Dockerfile # Frontend container
│
├── docker-compose.yml
├── requirements.txt
├── README.md
└── tests/                   # Unit tests (Pytest)


⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/Albin78/LLM-Project.git
cd LLM-Project

2. Install Dependencies
pip install -r requirements.txt

3. Run Backend (FastAPI)
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload


API available at: 👉 http://localhost:8000/docs

4. Run Frontend (Streamlit)
cd frontend
streamlit run streamlit_app.py


Frontend available at: 👉 http://localhost:8501

🐳 Docker Deployment

Build & run with Docker Compose:

docker-compose up --build


FastAPI → http://localhost:8000

Streamlit → http://localhost:8501

🔍 API Endpoints

POST /generate → Response from GPT model

POST /rag-query → Response with RAG pipeline

GET /health → Health check

Docs: 👉 http://localhost:8000/docs

📊 Workflow

User enters a medical query via Streamlit UI.

Query is sent to FastAPI backend.

RAG pipeline retrieves relevant documents from FAISS.

GPT-style model generates contextual response.

Answer is displayed in the frontend.

🚀 Future Work

Fine-tuning with LoRA for efficiency.

Expanding RAG knowledge base with PubMed & guidelines.

Adding conversation memory for multi-turn queries.

CI/CD with GitHub Actions


⚠️ Disclaimer

This project is for research and educational purposes only.
It is not a substitute for professional medical advice.
Always consult a doctor for medical concerns.

