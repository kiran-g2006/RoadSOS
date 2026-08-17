# RoadSOS – AI Road Safety RAG System

RoadSOS RAG is an AI-powered Retrieval-Augmented Generation (RAG) system designed to provide concise, evidence-based answers to questions related to road safety, motor vehicle laws, traffic regulations, and other official road-safety documents.

The system is developed as a separate backend module that can later be integrated with the main RoadSOS Flutter application developed by the project team.

The primary goal of this system is to allow users to ask natural-language road-safety questions and receive a concise answer based only on the legal and road-safety documents provided to the system.

---

# 1. Project Overview

RoadSOS is an AI-powered hybrid accident detection and emergency response platform.

As an additional intelligent feature, this project introduces a Road Safety RAG Assistant.

The RAG Assistant allows users to ask questions such as:

- What are the rules regarding protective headgear?
- What must a helmet have?
- What are the requirements for obtaining a driving licence?
- What are the rules for wearing protective headgear?
- What does Section 129 say?
- What are the road safety requirements mentioned in the Motor Vehicles Act?

Instead of relying entirely on the general knowledge of an AI model, the system retrieves relevant information from a collection of official road-safety documents and uses that evidence to generate the final answer.

This makes the system more suitable for legal and road-safety information retrieval because the answer is grounded in the documents provided to the system.

---

# 2. Main Objective

The main objectives of the RoadSOS RAG system are:

1. Process official road-safety and motor-vehicle documents.
2. Extract meaningful legal text from PDF documents.
3. Divide large documents into searchable chunks.
4. Convert document chunks into vector embeddings.
5. Store the embeddings in a FAISS vector database.
6. Retrieve relevant legal information for a user query.
7. Rerank retrieved results using a Cross-Encoder.
8. Filter weak or irrelevant evidence.
9. Group related legal provisions.
10. Generate a concise answer using a Generative AI model.
11. Provide the relevant section and page as evidence.
12. Expose the RAG system through a REST API.
13. Integrate the API with the RoadSOS Flutter application.

---

# 3. Why RAG is Used

A conventional chatbot may generate answers from the knowledge contained in its language model.

This can create problems when answering questions related to laws and regulations because:

- Laws can be specific.
- Legal provisions can be lengthy.
- Similar rules may exist in different sections.
- The model may generate information that is not present in the source documents.
- Users need references to the actual legal provision.

RoadSOS uses Retrieval-Augmented Generation.

The workflow is:

```
User Question
       |
       v
Query Embedding
       |
       v
FAISS Retrieval
       |
       v
Candidate Legal Documents
       |
       v
Cross-Encoder Reranking
       |
       v
Evidence Filtering
       |
       v
Relevant Legal Context
       |
       v
Generative AI
       |
       v
Concise Answer + Sources
```

The generative model is instructed to answer using only the retrieved legal evidence.

---

# 4. Current Document Categories

The project organizes road-safety documents into separate categories.

```text
data/
└── raw/
    ├── central/
    ├── state/
    ├── supplementary/
    └── traffic/
```

## Central

Contains central government laws and regulations.

Examples:

* Motor Vehicles Act
* Motor Vehicles Amendment Acts
* Central Motor Vehicle Rules

## State

Contains state-specific motor vehicle rules and regulations.

Examples:

* State Motor Vehicle Rules
* State-specific traffic regulations

## Supplementary

Contains supporting road-safety material.

Examples:

* Government road-safety guidelines
* Road-safety manuals
* Official safety documents

## Traffic

Contains traffic-related documents.

Examples:

* Traffic rules
* Road signs
* Road markings
* Driving regulations

The system can be extended by adding additional documents to these directories.

---

# 5. System Architecture

The current RAG architecture consists of five major stages:

```text
                    ┌─────────────────────┐
                    │     User Question   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Query Normalization │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Sentence Transformer│
                    │     Embedding       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FAISS Search     │
                    │ Vector Retrieval    │
                    └──────────┬──────────┘
                               │
                         Candidates
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Cross-Encoder     │
                    │     Reranker        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Evidence Quality    │
                    │      Filtering      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Legal Provision     │
                    │ Grouping/Dedup      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Relevant Legal      │
                    │     Context         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Gemini Generative │
                    │        Model        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Concise Answer +    │
                    │      Sources        │
                    └─────────────────────┘
```

---

# 6. Complete RAG Workflow

## Step 1 – Document Collection

Official road-safety documents are added to:

```text
data/raw/
```

They are organized into:

```text
central/
state/
supplementary/
traffic/
```

---

## Step 2 – PDF Text Extraction

The `ingest.py` module uses PyMuPDF to extract text from PDF documents.

Each PDF is processed page by page.

Example:

```text
PDF
 |
 ├── Page 1
 ├── Page 2
 ├── Page 3
 ├── ...
 └── Page N
```

Only pages containing text are processed.

---

## Step 3 – Text Cleaning

Extracted text is cleaned by:

* Replacing line breaks with spaces
* Removing unnecessary whitespace
* Normalizing spacing
* Removing spaces before punctuation

The objective is to preserve the legal meaning while creating cleaner text for retrieval.

---

## Step 4 – Legal Section Detection

The ingestion pipeline attempts to identify numbered legal sections.

For example:

```text
8. Grant of learner's licence
9. Grant of driving licence
10. ...
129. Wearing of protective headgear
```

The section number is stored as metadata.

---

## Step 5 – Chunking

Large legal sections are divided into smaller overlapping chunks.

The current implementation uses:

```text
Maximum words: 450
Overlap: 80 words
```

The overlap helps preserve context between neighboring chunks.

---

## Step 6 – Processed Dataset

The processed chunks are stored in:

```text
data/processed/chunks.json
```

Each chunk contains information similar to:

```json
{
    "id": 12,
    "text": "Legal text...",
    "source": "motor_vehicles_act_1988.pdf",
    "page": 13,
    "section": "9"
}
```

This metadata allows the system to identify where retrieved information came from.

---

# 7. Vector Database Creation

The `build_vectorstore.py` module converts document chunks into vector embeddings.

The embedding model currently used is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model produces 384-dimensional embeddings.

Example:

```text
Legal Text
    |
    v
Sentence Transformer
    |
    v
384-dimensional vector
```

These vectors are normalized and stored in a FAISS index.

---

# 8. FAISS Vector Search

FAISS is used for efficient similarity search.

The current implementation uses:

```text
IndexFlatIP
```

with normalized vectors.

Because the embeddings are normalized, inner-product similarity can be used as a cosine-similarity-style retrieval measure.

The vector database contains:

```text
vectorstore/
├── road_safety.index
└── metadata.json
```

`road_safety.index` contains the FAISS vectors.

`metadata.json` contains the corresponding document information.

---

# 9. Retrieval Pipeline

When the user asks a question:

```text
What must a helmet have?
```

the question is converted into an embedding.

FAISS searches the vector database for semantically similar chunks.

The system initially retrieves a broader candidate set.

Current retrieval configuration:

```text
top_k = 15
```

This gives the reranker more candidates to evaluate.

---

# 10. Cross-Encoder Reranking

The retrieved candidates are then passed through a Cross-Encoder.

Current model:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The Cross-Encoder evaluates:

```text
Question + Retrieved Legal Text
```

and produces a relevance score.

The candidates are then sorted according to their reranking scores.

The current reranking stage selects the strongest candidates before evidence filtering.

---

# 11. Evidence Quality Filtering

The system does not blindly send every retrieved document to the generative model.

Weak or irrelevant evidence is filtered.

This is important because vector similarity alone does not guarantee that a passage actually answers the question.

The system therefore performs an evidence quality check before answer generation.

---

# 12. Legal Provision Grouping

Multiple retrieved chunks may belong to the same legal provision.

For example:

```text
Section 129 – Page 69
Section 129 – Page 4
Section 129 – Amendment document
```

These can represent the same or related legal provision.

The system groups related provisions so that the final answer does not unnecessarily repeat the same rule.

---

# 13. Evidence Selection

After filtering and grouping, the system selects the strongest legal evidence.

The goal is to provide the generative model with a small amount of high-quality context rather than a large amount of unrelated information.

The system can return a maximum of a small number of strong provisions.

---

# 14. Answer Generation

The final evidence is passed to the Gemini generative model.

The model receives:

```text
User Question
+
Relevant Legal Evidence
```

The model is instructed to:

* Provide one direct answer.
* Use only the supplied evidence.
* Avoid outside knowledge.
* Avoid inventing laws.
* Avoid inventing penalties.
* Avoid irrelevant information.
* Mention the relevant section.
* Mention the source page where available.
* Keep the answer concise.
* Avoid repeating the same requirement.
* State when the information is not available in the supplied documents.

---

# 15. Example Query

User:

```text
What must a helmet have?
```

The system retrieves Section 129.

The final answer can be:

```text
According to Section 129, a protective helmet must be designed
to provide reasonable protection from injury and must have straps
or other fastenings so it can be securely fastened to the wearer's
head.
```

The response also contains source information such as:

```json
{
    "section": "129",
    "page": 69,
    "source": "motor_vehicles_act_1988.pdf"
}
```

---

# 16. RAG Response Structure

The system returns structured JSON.

Example:

```json
{
    "question": "What must a helmet have?",
    "answer": "According to Section 129, a protective helmet must be designed to provide reasonable protection from injury and must have straps or other fastenings so it can be securely fastened to the wearer's head.",
    "sources": [
        {
            "section": "129",
            "page": 69,
            "source": "motor_vehicles_act_1988.pdf",
            "retrieval_score": 0.6506,
            "rerank_score": 4.7609
        }
    ]
}
```

The scores are retained as backend evidence metadata and can be hidden from normal Flutter users.

---

# 17. Project Structure

```text
RoadSOS-RAG/
│
├── data/
│   │
│   ├── raw/
│   │   ├── central/
│   │   ├── state/
│   │   ├── supplementary/
│   │   └── traffic/
│   │
│   └── processed/
│       └── chunks.json
│
├── vectorstore/
│   ├── road_safety.index
│   └── metadata.json
│
├── src/
│   ├── ingest.py
│   ├── build_vectorstore.py
│   ├── rag.py
│   └── main.py
│
├── .env                  ← LOCAL ONLY, NOT GITHUB
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 18. Description of Source Files

## `src/ingest.py`

Responsible for:

* PDF discovery
* PDF text extraction
* Text cleaning
* Section detection
* Chunk creation
* Metadata generation

Output:

```text
data/processed/chunks.json
```

---

## `src/build_vectorstore.py`

Responsible for:

* Loading processed chunks
* Loading the embedding model
* Generating embeddings
* Normalizing vectors
* Creating the FAISS index
* Saving the vector database
* Saving metadata

Outputs:

```text
vectorstore/road_safety.index
vectorstore/metadata.json
```

---

## `src/rag.py`

Contains the main RAG pipeline.

Responsibilities include:

```text
Query
 ↓
Normalization
 ↓
Embedding
 ↓
FAISS retrieval
 ↓
Cross-Encoder reranking
 ↓
Evidence filtering
 ↓
Legal provision grouping
 ↓
Evidence selection
 ↓
Gemini answer generation
 ↓
Structured JSON response
```

It can also be executed directly for testing.

---

## `src/main.py`

Provides the FastAPI backend.

Its purpose is to expose the RAG system through HTTP so that the Flutter application can communicate with it.

Expected architecture:

```text
Flutter
   |
   | POST request
   v
FastAPI
   |
   v
rag.ask()
   |
   v
RAG pipeline
   |
   v
JSON response
```

---

# 19. Technology Stack

| Component              | Technology            |
| ----------------------- | ---------------------- |
| Programming Language   | Python                |
| API Framework          | FastAPI               |
| API Server              | Uvicorn               |
| PDF Processing          | PyMuPDF               |
| Embedding Model         | Sentence Transformers |
| Embedding Model         | all-MiniLM-L6-v2      |
| Vector Database          | FAISS                 |
| Vector Search           | IndexFlatIP           |
| Reranker                 | Cross-Encoder         |
| Reranker Model          | ms-marco-MiniLM-L6-v2 |
| Generative AI            | Gemini                |
| Environment Management  | python-dotenv         |
| Numerical Processing    | NumPy                 |
| ML Utilities             | Scikit-learn          |
| Frontend Integration     | Flutter REST API      |

---

# 20. Installation Requirements

## Python

Recommended:

```text
Python 3.10+
```

The current development environment uses Python 3.10.

---

# 21. Create Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell activation is restricted, the virtual environment can also be activated through the VS Code terminal or using the appropriate shell activation command.

---

# 22. Install Dependencies

Install the packages from:

```text
requirements.txt
```

Command:

```powershell
pip install -r requirements.txt
```

---

# 23. Environment Variables

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_gemini_api_key
```

The `.env` file must NOT be committed to GitHub.

Add it to `.gitignore`:

```text
.env
```

---

# 24. Add Road-Safety Documents

Place PDF files inside the appropriate directories.

Example:

```text
data/raw/
│
├── central/
│   ├── motor_vehicles_act_1988.pdf
│   └── motor_vehicles_amendment_act.pdf
│
├── state/
│   └── state_motor_vehicle_rules.pdf
│
├── supplementary/
│   └── road_safety_guidelines.pdf
│
└── traffic/
    └── traffic_rules.pdf
```

---

# 25. Process the Documents

Run:

```powershell
python -m src.ingest
```

Expected output:

```text
Processing: motor_vehicles_act_1988.pdf

Pages with text: 111

============================================================
LEGAL DOCUMENT PROCESSING COMPLETE
============================================================
Total chunks: ...
Output: data\processed\chunks.json
```

---

# 26. Build the Vector Database

After ingestion:

```powershell
python -m src.build_vectorstore
```

Expected workflow:

```text
Load chunks
     ↓
Load embedding model
     ↓
Generate embeddings
     ↓
Normalize embeddings
     ↓
Create FAISS index
     ↓
Save vector database
```

Expected output:

```text
vectorstore/
├── road_safety.index
└── metadata.json
```

---

# 27. Test the RAG System

Run:

```powershell
python -m src.rag
```

The system will ask:

```text
Enter your road-safety question:
```

Example:

```text
What are the rules regarding protective headgear?
```

The system retrieves relevant evidence and generates the answer.

---

# 28. Start the FastAPI Server

Run:

```powershell
uvicorn src.main:app --reload
```

The server runs at:

```text
http://127.0.0.1:8000
```

For local development, the server can be tested through the FastAPI documentation interface.

```text
http://127.0.0.1:8000/docs
```

---

# 29. API Workflow

The intended API workflow is:

```text
POST /api/rag/ask
```

Request:

```json
{
    "question": "What are the rules regarding protective headgear?"
}
```

The backend:

```text
Receive Question
       ↓
Normalize Question
       ↓
Generate Query Embedding
       ↓
FAISS Search
       ↓
Retrieve Candidates
       ↓
Cross-Encoder Reranking
       ↓
Evidence Quality Filtering
       ↓
Group Related Provisions
       ↓
Select Strong Evidence
       ↓
Generate Answer
       ↓
Return JSON
```

---

# 30. Flutter Integration

The RAG system is designed as a separate backend service.

The main RoadSOS Flutter application can communicate with it using HTTP.

Architecture:

```text
┌─────────────────────────────────────────┐
│             RoadSOS Flutter             │
│                                         │
│  Road Safety Assistant Screen           │
│              │                          │
│              ▼                          │
│       RAG Service / HTTP Client         │
└──────────────┬──────────────────────────┘
               │
               │ HTTP POST
               ▼
┌─────────────────────────────────────────┐
│              FastAPI Backend            │
│                                         │
│              RAG API                    │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│              RAG Pipeline               │
│                                         │
│ Query Embedding                         │
│ FAISS Retrieval                         │
│ Cross-Encoder Reranking                 │
│ Evidence Filtering                      │
│ Gemini Generation                       │
└─────────────────────────────────────────┘
```

The Gemini API key remains on the backend and should never be embedded in the Flutter application.

---

# 31. Development Workflow

The complete development workflow is:

```text
                 ROAD SAFETY PDFs
                        │
                        ▼
                  PDF INGESTION
                        │
                        ▼
                  TEXT CLEANING
                        │
                        ▼
                 SECTION DETECTION
                        │
                        ▼
                    CHUNKING
                        │
                        ▼
                chunks.json
                        │
                        ▼
              SENTENCE TRANSFORMER
                        │
                        ▼
                  EMBEDDINGS
                        │
                        ▼
                     FAISS
                        │
                        ▼
                VECTOR DATABASE
                        │
                        │
             USER ASKS QUESTION
                        │
                        ▼
                QUERY EMBEDDING
                        │
                        ▼
                 FAISS RETRIEVAL
                        │
                        ▼
               CANDIDATE RESULTS
                        │
                        ▼
              CROSS-ENCODER
                RERANKING
                        │
                        ▼
             EVIDENCE FILTERING
                        │
                        ▼
           PROVISION GROUPING
                        │
                        ▼
             PRIMARY EVIDENCE
                        │
                        ▼
                 GEMINI LLM
                        │
                        ▼
              CONCISE ANSWER
                        │
                        ▼
               JSON RESPONSE
                        │
                        ▼
               FASTAPI REST API
                        │
                        ▼
                ROADsOS FLUTTER
```

---

# 32. Current Capabilities

The current implementation supports:

* PDF-based knowledge ingestion
* Multiple road-safety document categories
* Legal section extraction
* Overlapping text chunking
* Semantic vector retrieval
* FAISS vector search
* Cross-Encoder reranking
* Evidence quality filtering
* Legal provision grouping
* Duplicate reduction
* Concise answer generation
* Source attribution
* Page references
* Section references
* REST API integration through FastAPI

---

# 33. Example Use Cases

### Helmet Regulations

```text
Question:
What are the rules regarding protective headgear?
```

The system retrieves Section 129 and provides a concise answer.

---

### Driving Licence

```text
Question:
What are the requirements for getting a driving licence?
```

The system searches relevant provisions and generates an answer based on the available documents.

---

### Legal Section Query

```text
Question:
What does Section 129 say?
```

The system retrieves the corresponding legal provision.

---

### Safety Requirements

```text
Question:
What safety requirements are mentioned for motorcycles?
```

The system searches the road-safety knowledge base and returns the strongest relevant evidence.

---

# 34. Important Design Principle

The system follows an evidence-grounded answering strategy.

The generative model is instructed:

```text
Use ONLY the provided legal evidence.
```

If sufficient evidence cannot be found, the system returns:

```text
The information was not found in the provided
road-safety documents.
```

This reduces unsupported answers and helps prevent the model from inventing legal information.

---

# 35. Security

The Gemini API key must never be committed to GitHub.

Use:

```text
.env
```

and add:

```text
.env
```

to `.gitignore`.

Do not place:

```text
GEMINI_API_KEY
```

inside Flutter source code.

The intended architecture is:

```text
Flutter
   |
   | Question
   ▼
FastAPI
   |
   | API key
   ▼
Gemini
```

---

# 36. GitHub Files

The repository should contain:

```text
src/
requirements.txt
README.md
.gitignore
```

Generated files such as the virtual environment and secret configuration should not be committed.

Recommended `.gitignore`:

```gitignore
venv/
.venv/

.env

__pycache__/
*.pyc

.vscode/
.idea/

data/processed/
vectorstore/

*.log

.DS_Store
Thumbs.db
```

If the source PDFs are legally redistributable and you want the repository to be self-contained, they can be added separately. Otherwise, keep them out of Git and document the required data sources.

Note: `vectorstore/` and `data/processed/` are generated artifacts and are excluded from Git — regenerate them locally by following the ingestion and vector database build steps above.

---

# 37. Future Integration with RoadSOS

The RAG Assistant is intended to become one component of the larger RoadSOS application.

The overall RoadSOS platform can eventually follow this architecture:

```text
                         ROADsOS
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
       Accident         Emergency       Road Safety
       Detection        Response        Assistant
             │              │              │
             ▼              ▼              ▼
          Sensors         SMS/GPS         RAG API
             │              │              │
             ▼              ▼              ▼
            ML           Offline        FastAPI
          Model          System           │
                                          ▼
                                       FAISS
                                          │
                                          ▼
                                       Gemini
```

The RAG component does not replace the accident detection system.

It provides a separate intelligent knowledge-assistance capability for road-safety and legal questions.

---

# 38. Future Improvements

Possible future improvements include:

* Hybrid keyword + semantic retrieval
* Better legal-document section detection
* Metadata-aware retrieval
* Document priority ranking
* State-specific rule filtering
* Multilingual road-safety questions
* Conversation history
* Voice-based road-safety assistant
* Streaming responses
* Improved citation generation
* Administrative document management
* Automatic document updates
* Better evaluation benchmarks
* Production deployment
* Integration with the RoadSOS Flutter application

---

# 39. Project Status

Current status:

```text
[✓] Road-safety document collection
[✓] PDF ingestion
[✓] Text extraction
[✓] Text cleaning
[✓] Legal section detection
[✓] Chunking
[✓] Embedding generation
[✓] FAISS vector database
[✓] Semantic retrieval
[✓] Cross-Encoder reranking
[✓] Evidence filtering
[✓] Provision grouping
[✓] Concise answer generation
[✓] Source metadata
[✓] RAG testing
[✓] FastAPI backend
[ ] Flutter API integration
[ ] Production deployment
```

---

# 40. Conclusion

RoadSOS RAG provides an evidence-grounded AI interface for road-safety and motor-vehicle legal information.

The system combines:

* Document processing
* Semantic embeddings
* FAISS vector retrieval
* Cross-Encoder reranking
* Evidence filtering
* Legal provision grouping
* Generative AI

to transform official road-safety documents into an interactive question-answering system.

The backend is intentionally separated from the main Flutter application, allowing it to be independently developed, tested, deployed, and eventually integrated into the complete RoadSOS platform through a REST API.

The final objective is to provide RoadSOS users with a reliable road-safety assistant capable of answering questions using the project's approved road-safety knowledge base.
