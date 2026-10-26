# Financial Analyst — RAG-Powered Financial Analysis Assistant

**An AI-powered financial analysis assistant built with Retrieval-Augmented Generation (RAG), LangChain, Google Gemini, and Qdrant.**

## Overview

Financial Analyst is a Generative AI project designed to simplify financial document analysis.

Financial reports often contain hundreds of pages, making it time-consuming to retrieve financial indicators, compare historical performance, and identify relevant information.

Our solution combines **semantic search, Large Language Models (LLMs), and Python-based financial calculations** to provide answers grounded in real corporate financial reports, with references to the original documents.

## Key Features

- **Financial Question Answering:** Ask natural-language questions about companies and their financial reports.
- **Semantic Document Retrieval:** Retrieve relevant information using multilingual vector embeddings.
- **Financial Ratio Analysis:** Calculate gross margin, operating margin, net margin, and Return on Equity (ROE).
- **Growth Analysis:** Compare financial performance across reporting periods.
- **SWOT Analysis:** Identify strengths, weaknesses, opportunities, and threats based on retrieved documents.
- **Source Traceability:** Associate financial information with its source document and page.
- **Validation Controls:** Apply Python-based checks to extracted values and citations.

## System Architecture

The system follows a RAG pipeline organized into two main stages.

### 1. Document Processing and Indexing

**FinanceBench PDFs → Text Extraction → Cleaning → Chunking → Embeddings → Qdrant Cloud**

- Collect financial reports from FinanceBench.
- Extract text page by page using `pypdf`.
- Clean and split documents with `RecursiveCharacterTextSplitter`.
- Preserve metadata, including company, year, report type, document, and page.
- Generate 1,024-dimensional embeddings using `BAAI/bge-m3`.
- Store document chunks and vectors in Qdrant Cloud.

**Processing configuration:**

| Parameter | Value |
|---|---|
| Documents selected | 229 |
| Generated chunks | 90,456 |
| Chunk size | 1,500 characters |
| Chunk overlap | 150 characters |
| Embedding model | BAAI/bge-m3 |
| Vector dimensions | 1,024 |
| Vector database | Qdrant Cloud |
| Similarity metric | Cosine |

### 2. Financial Question Answering

**User Question → Semantic Retrieval → Gemini → Python Validation & Calculations → Sourced Answer**

When a user submits a question:

1. The system retrieves relevant financial passages from Qdrant.
2. Gemini extracts the required financial information.
3. Python checks numerical values and source references.
4. Financial calculations are performed programmatically when required.
5. The system generates a response with supporting document references.

The validation layer helps identify inconsistencies, but does not guarantee that every extracted value is correct.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| RAG Framework | LangChain |
| Generative AI | Google Gemini |
| Embedding Model | BAAI/bge-m3 |
| Vector Database | Qdrant Cloud |
| PDF Processing | pypdf |
| Text Chunking | RecursiveCharacterTextSplitter |
| Structured Output | Pydantic |
| Dataset | FinanceBench |

## Dataset

The project uses **FinanceBench**, a benchmark based on real financial documents from publicly traded companies.

According to the project dataset summary:

- 361 original financial documents
- 40 companies
- 229 documents selected for processing
- 90,456 text chunks generated

The collection includes annual reports, quarterly filings, 8-K filings, and earnings documents.

## Retrieval Evaluation

The initial retrieval evaluation was conducted using 14 locally selected financial questions.

| Metric | Initial Result |
|---|---|
| Questions evaluated | 14 |
| Hit@8 | 78.6% |
| Top-1 page retrieval | 50.0% |
| Mean Reciprocal Rank (MRR) | 0.589 |

These results evaluate **document retrieval**, not the accuracy of final generated answers. A larger benchmark evaluation is needed to assess generalization.

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file based on `.env.example` and configure:

```env
QDRANT_URL=
QDRANT_API_KEY=
GOOGLE_API_KEY=
GEMINI_MODEL=
```

Keep API keys private and never commit the `.env` file.

## Project Structure

```text
financial-analyst/
├── src/
├── rag_eval/
├── financial-chat-ui/
├── indexation_colab.ipynb
├── indexation_kaggle.ipynb
├── requirements.txt
├── .env.example
└── README.md
```

## Limitations and Future Improvements

- Improve financial table extraction and numerical consistency.
- Strengthen validation of financial values, years, and units.
- Experiment with hybrid search and reranking.
- Extend retrieval and answer-generation evaluation.
- Improve reliability and usability of the financial analysis interface.

## Project Team

Developed as a Generative AI academic project by:

- **Meryem Elbiach**
- **Hind Elharrab**
- **Basma Elgharbaoui**
- **Hajar Braidi**

**Academic Supervisor:** Ibtissam Abnane

---

**Financial Analyst — Turning financial documents into accessible, evidence-based insights.**