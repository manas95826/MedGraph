# MedGraph — Medical Research Knowledge Explorer

A full-stack **GraphRAG** (Graph Retrieval-Augmented Generation) system built on top of 30 synthetic medical research papers. It combines a structured knowledge graph (Neo4j), dense vector search (Qdrant), and GPT-4o-mini to let you explore, search, and ask questions over the dataset.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (index.html)                     │
│   Dashboard │ Graph Explorer │ Semantic Search │ Ask the Graph  │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP (REST)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FastAPI  backend/api.py  :8000                  │
│                                                                 │
│  GET  /graph/nodes          → entity nodes for graph explorer   │
│  GET  /graph/edges          → relationships                     │
│  GET  /graph/stats          → dashboard charts                  │
│  GET  /graph/compound/{n}   → compound subgraph                 │
│  GET  /graph/timeline       → papers per year + efficacy        │
│  POST /search               → Qdrant semantic search            │
│  POST /qa                   → Hybrid GraphRAG answer            │
└────────┬──────────────────────┬────────────────────────────────┘
         │                      │
         ▼                      ▼
┌────────────────┐   ┌──────────────────────────────────────────┐
│  Neo4j  :7687  │   │  Qdrant  :6335 (host) → :6333 (internal) │
│                │   │                                          │
│  Typed nodes:  │   │  Collections:                            │
│  - Paper       │   │  • medgraph_chunks  ← API semantic search│
│  - Compound    │   │  • DocumentChunk_text  ┐                 │
│  - Disease     │   │  • Entity_name         ├ Cognee internal │
│  - Researcher  │   │  • TextSummary_text    ┘                 │
│  - Institution │   │                                          │
│  - Journal     │   │  Embeddings: all-MiniLM-L6-v2 (384-dim) │
│                │   │  Similarity: cosine                      │
│  Typed edges:  │   └──────────────────────────────────────────┘
│  STUDIES       │
│  TARGETS       │   ┌──────────────────────────────────────────┐
│  TESTED_AGAINST│   │  Cognee 0.2.1  (ingest only)             │
│  AUTHORED      │   │                                          │
│  AFFILIATED_WITH│  │  vector backend → Qdrant (above)         │
│  CONDUCTED_AT  │   │  graph backend  → Neo4j (above)          │
│  COLLABORATED  │   │  LLM → GPT-4o-mini entity extraction     │
│  PUBLISHED_IN  │   │  ENV=dev required (adapter bug workaround│
└────────────────┘   └──────────────────────────────────────────┘
```

---

## Data Model

The knowledge graph holds **6 node types** and **7 relationship types**:

```
(Researcher)-[:AUTHORED]----------→(Paper)
(Researcher)-[:AFFILIATED_WITH]---→(Institution)
(Paper)-----[:CONDUCTED_AT]-------→(Institution)
(Paper)-----[:COLLABORATED_WITH]--→(Institution)   ← collaborating inst.
(Paper)-----[:STUDIES]------------→(Compound)
(Paper)-----[:TARGETS]------------→(Disease)
(Paper)-----[:PUBLISHED_IN]-------→(Journal)
(Compound)--[:TESTED_AGAINST {efficacy}]→(Disease)
```

Each `Paper` node stores: `id`, `title`, `year`, `method`, `efficacy` (%).
Each `Compound` node stores: `name`, `type` (antimicrobial / antiviral / anticancer / anti-inflammatory / antifungal / neuroprotective).
The `TESTED_AGAINST` edge stores `efficacy` directly so you can query it without hitting the paper.

---

## How Ingestion Works (`backend/ingest.py`)

The pipeline runs in three sequential steps. Order matters — Cognee's `prune_system` wipes both Qdrant and Neo4j, so the direct ingests run after it.

### Step 1 — Cognee LLM pipeline (entity extraction)
- Configures Cognee with Qdrant as its vector store (`cognee[qdrant]==0.2.1`) and Neo4j as its graph store
- Sets `ENV=dev` in the environment — the Qdrant adapter does `os.getenv("ENV").lower()` and crashes if unset
- Calls `prune_system(metadata=True)` — wipes all Qdrant collections and Neo4j data for a clean slate
- Feeds all 30 paper texts to `cognee.add()` as dataset `medical_papers`
- Calls `cognee.cognify()` which:
  - Splits each text into sentence-boundary chunks
  - Calls GPT-4o-mini to extract named entities and relationships per chunk
  - Embeds chunks and writes them to Qdrant (collections: `DocumentChunk_text`, `Entity_name`, `TextSummary_text`, etc.)
  - Writes extracted graph (Cognee's internal `__Node__` schema) to Neo4j

### Step 2 — Qdrant `medgraph_chunks` (semantic search collection)
- Runs after Cognee so `prune_system` doesn't wipe it
- Encodes each paper's `text` with `all-MiniLM-L6-v2` (384-dim, local CPU/MPS)
- Drops and recreates collection `medgraph_chunks`, upserts 30 vectors
- Payload per vector: `paper_id`, `title`, `year`, `compound`, `disease`, `institution`, `journal`, `method`, `efficacy`, full `text`
- This is the collection the API's `/search` and `/qa` endpoints query

### Step 3 — Neo4j typed schema (graph queries)
- Runs after Cognee so `prune_system` doesn't wipe it
- Clears Cognee's internal `__Node__` nodes, keeping only typed nodes
- Creates uniqueness constraints for all 6 node types
- Uses `MERGE` for each entity so re-runs are idempotent
- Writes all typed nodes and relationships from structured fields in `papers.json`
- Result: 89 nodes, ~200+ relationships available for precise Cypher traversal

---

## How Hybrid QA Works (`POST /qa`)

When you ask a question, two retrieval branches run in parallel and are fused into one LLM prompt:

```
User question: "Which compounds treat breast cancer?"
        │
        ├── Branch A: Semantic (Qdrant)
        │     1. Encode question → 384-dim vector
        │     2. cosine search in medgraph_chunks, threshold 0.25
        │     3. Returns top-5 paper text chunks with scores
        │
        ├── Branch B: Graph (Neo4j)
        │     1. Scan all Compound + Disease names in graph
        │     2. Check which ones appear literally in the question
        │     3a. If compound matched → Cypher:
        │         MATCH (c:Compound)-[r:TESTED_AGAINST]->(d)
        │         MATCH (p:Paper)-[:STUDIES]->(c)
        │         MATCH (res:Researcher)-[:AUTHORED]->(p)
        │         → returns compound, disease, efficacy %, researcher, year, method
        │     3b. If disease matched → Cypher:
        │         MATCH (d:Disease)<-[:TARGETS]-(p:Paper)
        │         MATCH (c:Compound)<-[:STUDIES]-(p)
        │         → returns compound, efficacy, year, sorted by efficacy DESC
        │     3c. If nothing matched → top-8 by efficacy across all data
        │
        └── Fuse → GPT-4o-mini
              system: medical research assistant persona
              user:   graph facts block + semantic evidence block + question
              → grounded answer with compound names, %, researchers cited
```

The response includes `graph_facts`, `semantic_sources`, and `retrieval_info` so the frontend can show exactly what was used.

---

## How Semantic Search Works (`POST /search`)

Pure vector search — no LLM involved:

1. Encode the query string with `all-MiniLM-L6-v2`
2. Call `qdrant.query_points()` with cosine similarity, threshold 0.3
3. Optional `year_filter` narrows results to a specific publication year
4. Returns ranked results with score, metadata, and a 300-char text snippet

---

## The Dataset (`data/papers.json`)

30 synthetic but realistic medical research papers generated by `scripts/generate_data.py`:

| Dimension | Values |
|-----------|--------|
| **Compounds** | GX-471, CMP-88, BRD-2201, NXP-334, ZLT-904, VKR-115, MDX-557, PLQ-772, TRF-090, GSK-441, AMR-623, HCV-309 |
| **Compound types** | antimicrobial, antiviral, anticancer, anti-inflammatory, antifungal, neuroprotective |
| **Diseases** | H. pylori, Hepatitis C, Lung cancer, Rheumatoid arthritis, MRSA, Candida albicans, Glioblastoma, SARS-CoV-2, Alzheimer's, Crohn's, Tuberculosis, Breast cancer, Pancreatic cancer, HIV, E. coli |
| **Researchers** | 12, across AIIMS, IIT Bombay, Peking Union, Johns Hopkins, Kyoto University, Charité Berlin, Karolinska, CMC Vellore, etc. |
| **Journals** | Lancet Infectious Diseases, Nature Medicine, NEJM, JAMA Oncology, Cancer Research, Cell Host & Microbe, etc. |
| **Study methods** | RCT, Phase II Trial, Retrospective Cohort, In Vitro, Mouse Xenograft, Single-Cell RNA-seq, CRISPR Screen, Proteomics, Molecular Docking, Systematic Review |
| **Efficacy range** | 30–97% (random per paper) |
| **Years** | 2018–2024 |

---

## Setup

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- OpenAI API key

### Step 1 — Start databases
```bash
docker compose up -d
# Neo4j:  http://localhost:7474  (browser UI)
# Qdrant: http://localhost:6335  (host-mapped port)
```

> **Note**: Qdrant's host port is **6335** (remapped from 6333 internally) to avoid conflicts. The `.env` and all backend code use `QDRANT_PORT=6335`.

### Step 2 — Python environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Configure
Edit `.env` and set your OpenAI API key:
```
OPENAI_API_KEY=sk-...
```

### Step 4 — Generate data
```bash
python scripts/generate_data.py
# Writes data/papers.json
```

### Step 5 — Ingest (5–10 min, LLM-bound)
```bash
python backend/ingest.py
```
Order of operations:
1. Cognee LLM extraction → Qdrant (internal collections) + Neo4j
2. Direct embed → Qdrant `medgraph_chunks` (for API semantic search)
3. Typed schema → Neo4j (for API graph queries)

### Step 6 — Start the API
```bash
cd backend && python api.py
# Serves frontend + API at http://localhost:8000
```

### Step 7 — Open
```
http://localhost:8000
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Returns `{neo4j, qdrant}` status |
| `GET` | `/graph/nodes` | All 89 entity nodes with type, label, color |
| `GET` | `/graph/edges` | All relationships (up to 500) |
| `GET` | `/graph/stats` | Node counts, top compounds/diseases/institutions, method breakdown, timeline |
| `GET` | `/graph/compound/{name}` | Subgraph around one compound — its papers, researchers, diseases |
| `GET` | `/graph/timeline` | Per-year data: compound, disease, efficacy for timeline chart |
| `POST` | `/search` | `{query, limit?, year_filter?}` → semantic search results |
| `POST` | `/qa` | `{question}` → `{answer, graph_facts, semantic_sources, retrieval_info}` |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Graph DB | Neo4j 5.18 Community |
| Vector DB | Qdrant (latest) |
| LLM pipeline | Cognee 0.2.1 with Qdrant adapter (entity extraction → Qdrant + Neo4j) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim) |
| LLM (QA) | OpenAI GPT-4o-mini |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla JS + D3.js v7 + Chart.js 4 (single `index.html`, no build step) |
| Infra | Docker Compose |

---

## Stopping

```bash
docker compose down       # stop containers, data preserved in Docker volumes
docker compose down -v    # stop + delete all volumes (full reset)
```
