# Legal QA Chatbot (Knowledge-based, Retrieval-Augmented)

## Project Overview

This project implements a Legal Question-Answering Chatbot focused on the Consumer Protection Act (CPA) 2019. The chatbot is designed to provide relevant legal information, classify user complaints by intent, and offer guidance on forums and documentation, all while presenting information in plain language.

## Key Features

- **Knowledge-based Retrieval**: Answers legal queries by retrieving relevant sections from the CPA 2019.
- **Intent Classification**: Automatically identifies the intent behind user complaints (e.g., 'battery issue', 'delivery delay', 'refund issue', 'camera issue').
- **Automated Fallback Mechanism**: If the primary intent classifier has low confidence, the system dynamically falls back to HDBSCAN clustering and a similarity search against known complaint patterns to provide the most relevant intent classification, eliminating the need for manual review.
- **Plain Language Summaries**: Utilizes a small Language Model (LLM) to summarize complex legal text into easily understandable plain language.
- **Rule Engine**: Advises on eligibility and recommended consumer forums based on complaint details (e.g., price, date).
- **Modular Design**: Separates concerns into distinct modules for PDF processing, embedding, retrieval, rule engine, LLM summarization, complaint analysis, and intent prediction.

## Technical Components

### Backend (FastAPI - Python)

- **Text Ingestion**: Converts PDF legal documents (CPA 2019) into structured, clean text.
- **Sentence Embeddings**: Uses Sentence Transformers (`all-MiniLM-L6-v2`) to convert text (legal sections, complaints, queries) into dense numerical vectors.
- **Vector Database**: Employs FAISS for efficient similarity search to retrieve relevant legal sections.
- **Intent Classification**: A `LogisticRegression` model is trained on a dataset of labeled complaints (`complaints_data.json`) to classify user queries by intent.
- **HDBSCAN Clustering**: Used to identify natural clusters within complaint data, providing a fallback mechanism for new or ambiguous intents.
- **FastAPI**: Provides RESTful API endpoints for chat interaction, complaint analysis, and a continuous learning mechanism.
- **Uvicorn**: ASGI server to run the FastAPI application.

### Frontend (React - JavaScript)

- A user-friendly chat interface for interacting with the Legal QA Chatbot.
- Communicates with the FastAPI backend via `axios`.

## Setup and Installation

### Prerequisites

- Python 3.8+ (for backend)
- Node.js and npm (for frontend)
- Git

### Backend Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/aruk04/legal_chatbot.git
    cd legal_chatbot
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Process the legal document and train models (initial setup):**
    _Make sure `ConsumerProtection Act 2019.pdf` is in the project root._ (If you don't have it, a placeholder empty file will be created by `pdf_converter.py`)

    ```bash
    python pdf_converter.py
    python section_parser.py
    python embedder.py
    python retriever.py
    python create_complaints_json.py # Creates complaints_data.json with initial intents
    python complaint_analyzer.py # Trains LR and HDBSCAN models
    # You might want to run test cases in intent_predictor.py as well:
    # python intent_predictor.py
    ```

5.  **Start the FastAPI backend server:**
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    The backend will be accessible at `http://localhost:8000`.

### Frontend Setup

1.  **Navigate to the frontend directory:**

    ```bash
    cd frontend
    ```

2.  **Install Node.js dependencies:**

    ```bash
    npm install
    ```

3.  **Start the React development server:**
    ```bash
    npm start
    ```
    The frontend will typically open in your browser at `http://localhost:3000`.

## Usage

- Open your web browser to `http://localhost:3000`.
- Enter your legal queries related to the Consumer Protection Act 2019.
- Observe the retrieved sections, plain-language summaries, and the predicted intent of your query.

## Continuous Learning (Automated)

- The system automatically classifies new complaints. If a new type of complaint comes in that the primary classifier has low confidence in, it falls back to HDBSCAN clustering and a similarity search against all known patterns to provide the best possible classification.
- To add new labeled complaint data to improve the model's accuracy on specific intents, use the `/add_complaint` endpoint (e.g., via `curl` or a custom script) to update `complaints_data.json`.
  ```bash
  curl -X POST "http://localhost:8000/add_complaint" -H "Content-Type: application/json" -d "{\"complaint_text\":\"My phone's camera suddenly stopped working after the latest software update and shows a black screen.\",\"intent_label\":\"camera issue\"}"
  ```
  _Note: Adding new data via this endpoint automatically triggers model retraining and reloading._

## Project Structure

```
legal_chatbot/
├── .venv/                   # Python virtual environment
├── backend/                 # (Optional: if you decide to separate backend code)
├── frontend/                # React frontend application
│   ├── public/
│   ├── src/
│   └── package.json
├── complaint_analyzer.py    # HDBSCAN and Logistic Regression for intent classification
├── complaints_data.json     # Dataset for intent classification
├── create_complaints_json.py# Script to initialize/update complaints_data.json
├── cluster_intent_manager.py# Manages cluster-to-intent mappings and pending review (now automated)
├── consumer_protection_act_sections_clean.json # Cleaned legal sections
├── consumer_protection_act_sections.json     # Raw parsed legal sections
├── consumer_protection_act.txt               # Extracted text from PDF
├── ConsumerProtection Act 2019.pdf           # Original PDF document
├── cpa_faiss_index.bin      # FAISS index for legal sections
├── embedder.py              # Handles sentence embeddings
├── hdbscan_model.joblib     # Saved HDBSCAN model
├── intent_classifier.joblib # Saved Logistic Regression model
├── intent_predictor.py      # Main intent prediction logic with fallback
├── json_cleaner.py          # Helper for JSON cleaning
├── llm_agent.py             # LLM for summaries
├── main.py                  # FastAPI application entry point
├── pdf_converter.py         # Converts PDF to text
├── requirements.txt         # Python dependencies
├── retriever.py             # Retrieves relevant legal sections
├── rule_engine.py           # Implements business logic for forum/eligibility
├── section_parser.py        # Parses and chunks legal text
├── sections_map.json        # Mapping of sections for retrieval
└── README.md                # Project README
```
