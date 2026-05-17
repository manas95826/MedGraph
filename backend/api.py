"""
backend/api.py
──────────────
FastAPI server exposing:
  GET  /graph/nodes          → all entity nodes + types for the visual explorer
  GET  /graph/edges          → all relationships
  GET  /graph/stats          → counts, top compounds, top diseases, timeline
  GET  /graph/compound/{name}→ subgraph around a compound
  GET  /graph/timeline       → papers per year + efficacy trend
  POST /search               → hybrid Qdrant (semantic) search
  POST /qa                   → GraphRAG: hybrid Neo4j traversal + Qdrant + LLM answer
  GET  /health               → services health check

Run: uvicorn backend.api:app --reload --port 8000
"""

import os, json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI   = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER  = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS  = os.getenv("NEO4J_PASSWORD", "medgraph123")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6335))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "medgraph_chunks")

# ── Clients ───────────────────────────────────────────────────────────────────
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
qdrant       = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder     = SentenceTransformer("all-MiniLM-L6-v2")
llm          = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="MedGraph API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Serve the frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def neo4j_query(cypher: str, params: dict = {}) -> list[dict]:
    with neo4j_driver.session() as s:
        result = s.run(cypher, params)
        return [dict(r) for r in result]

COLOR_MAP = {
    "Paper":       "#4F8EF7",
    "Researcher":  "#34C78A",
    "Institution": "#F5A623",
    "Compound":    "#E05CF4",
    "Disease":     "#F74F4F",
    "Journal":     "#4FCFF5",
}

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "MedGraph API running", "docs": "/docs"}


@app.get("/health")
def health():
    status = {}
    try:
        neo4j_query("RETURN 1")
        status["neo4j"] = "ok"
    except Exception as e:
        status["neo4j"] = f"error: {e}"
    try:
        qdrant.get_collections()
        status["qdrant"] = "ok"
    except Exception as e:
        status["qdrant"] = f"error: {e}"
    return status


@app.get("/graph/nodes")
def graph_nodes():
    """All entity nodes for the force-directed graph."""
    rows = neo4j_query("""
        MATCH (n)
        WHERE n:Paper OR n:Researcher OR n:Institution
           OR n:Compound OR n:Disease OR n:Journal
        RETURN
            elementId(n) AS id,
            labels(n)[0]  AS type,
            CASE
                WHEN n:Paper       THEN n.title
                WHEN n:Researcher  THEN n.name
                WHEN n:Institution THEN n.name
                WHEN n:Compound    THEN n.name
                WHEN n:Disease     THEN n.name
                WHEN n:Journal     THEN n.name
                ELSE 'Unknown'
            END AS label,
            CASE WHEN n:Compound THEN n.type ELSE null END AS subtype,
            CASE WHEN n:Paper    THEN n.year ELSE null END AS year,
            CASE WHEN n:Paper    THEN n.efficacy ELSE null END AS efficacy
    """)

    nodes = []
    for r in rows:
        nodes.append({
            "id":      r["id"],
            "label":   r["label"],
            "type":    r["type"],
            "subtype": r.get("subtype"),
            "year":    r.get("year"),
            "efficacy":r.get("efficacy"),
            "color":   COLOR_MAP.get(r["type"], "#999"),
        })
    return {"nodes": nodes, "count": len(nodes)}


@app.get("/graph/edges")
def graph_edges():
    """All relationships for the graph."""
    rows = neo4j_query("""
        MATCH (a)-[r]->(b)
        WHERE (a:Paper OR a:Researcher OR a:Institution OR a:Compound OR a:Disease OR a:Journal)
          AND (b:Paper OR b:Researcher OR b:Institution OR b:Compound OR b:Disease OR b:Journal)
        RETURN
            elementId(a) AS source,
            elementId(b) AS target,
            type(r)       AS rel,
            CASE WHEN r.efficacy IS NOT NULL THEN r.efficacy ELSE null END AS weight
        LIMIT 500
    """)
    return {"edges": rows, "count": len(rows)}


@app.get("/graph/stats")
def graph_stats():
    """Dashboard stats: node counts, top compounds, top diseases, method breakdown."""

    counts = neo4j_query("""
        MATCH (n)
        RETURN labels(n)[0] AS type, count(n) AS cnt
        ORDER BY cnt DESC
    """)

    top_compounds = neo4j_query("""
        MATCH (c:Compound)-[:TESTED_AGAINST]->(d:Disease)
        WITH c, count(d) AS studies, avg(toFloat(0)) AS avg_eff
        MATCH (c)-[r:TESTED_AGAINST]->(d2)
        WITH c, count(d2) AS studies, avg(toFloat(r.efficacy)) AS avg_eff
        RETURN c.name AS compound, c.type AS type,
               studies, round(avg_eff, 1) AS avg_efficacy
        ORDER BY studies DESC LIMIT 8
    """)

    top_diseases = neo4j_query("""
        MATCH (p:Paper)-[:TARGETS]->(d:Disease)
        RETURN d.name AS disease, count(p) AS papers
        ORDER BY papers DESC LIMIT 8
    """)

    top_institutions = neo4j_query("""
        MATCH (p:Paper)-[:CONDUCTED_AT]->(i:Institution)
        RETURN i.name AS institution, count(p) AS papers
        ORDER BY papers DESC LIMIT 8
    """)

    methods = neo4j_query("""
        MATCH (p:Paper)
        RETURN p.method AS method, count(p) AS cnt
        ORDER BY cnt DESC
    """)

    timeline = neo4j_query("""
        MATCH (p:Paper)
        RETURN p.year AS year, count(p) AS papers, avg(toFloat(p.efficacy)) AS avg_eff
        ORDER BY year
    """)

    return {
        "node_counts":        counts,
        "top_compounds":      top_compounds,
        "top_diseases":       top_diseases,
        "top_institutions":   top_institutions,
        "methods":            methods,
        "timeline":           timeline,
    }


@app.get("/graph/compound/{name}")
def compound_subgraph(name: str):
    """Subgraph centered on a compound — papers, researchers, diseases."""
    nodes_raw = neo4j_query("""
        MATCH (c:Compound {name: $name})
        OPTIONAL MATCH (c)<-[:STUDIES]-(p:Paper)
        OPTIONAL MATCH (p)<-[:AUTHORED]-(r:Researcher)
        OPTIONAL MATCH (p)-[:TARGETS]->(d:Disease)
        WITH collect(DISTINCT c) + collect(DISTINCT p) +
             collect(DISTINCT r) + collect(DISTINCT d) AS all_nodes
        UNWIND all_nodes AS n
        RETURN DISTINCT
            elementId(n) AS id,
            labels(n)[0]  AS type,
            CASE
                WHEN n:Compound    THEN n.name
                WHEN n:Paper       THEN n.title
                WHEN n:Researcher  THEN n.name
                WHEN n:Disease     THEN n.name
            END AS label
    """, {"name": name})

    edges_raw = neo4j_query("""
        MATCH (c:Compound {name: $name})
        OPTIONAL MATCH (c)<-[r1:STUDIES]-(p:Paper)
        OPTIONAL MATCH (p)<-[r2:AUTHORED]-(res:Researcher)
        OPTIONAL MATCH (p)-[r3:TARGETS]->(d:Disease)
        WITH collect(r1) + collect(r2) + collect(r3) AS rels
        UNWIND rels AS r
        RETURN elementId(startNode(r)) AS source,
               elementId(endNode(r))   AS target,
               type(r)                  AS rel
    """, {"name": name})

    return {
        "nodes": [{"id": n["id"], "label": n["label"],
                   "type": n["type"], "color": COLOR_MAP.get(n["type"], "#999")}
                  for n in nodes_raw if n["id"]],
        "edges": [e for e in edges_raw if e["source"] and e["target"]],
    }


@app.get("/graph/timeline")
def timeline():
    return neo4j_query("""
        MATCH (p:Paper)-[:STUDIES]->(c:Compound)-[r:TESTED_AGAINST]->(d:Disease)
        RETURN p.year AS year, c.name AS compound,
               d.name AS disease, r.efficacy AS efficacy
        ORDER BY p.year
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Search — Qdrant semantic
# ─────────────────────────────────────────────────────────────────────────────
class SearchReq(BaseModel):
    query: str
    limit: int = 8
    year_filter: Optional[int] = None

@app.post("/search")
def semantic_search(req: SearchReq):
    """Dense vector search over paper chunks."""
    vec = embedder.encode(req.query).tolist()

    qfilter = None
    if req.year_filter:
        qfilter = Filter(must=[
            FieldCondition(key="year", match=MatchValue(value=req.year_filter))
        ])

    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=vec,
        limit=req.limit,
        query_filter=qfilter,
        with_payload=True,
        score_threshold=0.3,
    ).points

    return {
        "query": req.query,
        "results": [
            {
                "score":       round(h.score, 3),
                "paper_id":    h.payload["paper_id"],
                "title":       h.payload["title"],
                "year":        h.payload["year"],
                "compound":    h.payload["compound"],
                "disease":     h.payload["disease"],
                "institution": h.payload["institution"],
                "journal":     h.payload["journal"],
                "efficacy":    h.payload["efficacy"],
                "snippet":     h.payload["text"][:300] + "...",
            }
            for h in hits
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# QA — Hybrid GraphRAG
# ─────────────────────────────────────────────────────────────────────────────
class QAReq(BaseModel):
    question: str

@app.post("/qa")
def graphrag_qa(req: QAReq):
    """
    Hybrid GraphRAG:
      1. Semantic search in Qdrant → relevant chunks
      2. Graph traversal in Neo4j → structured facts
      3. Fuse both → LLM generates answer
    """
    question = req.question

    # Branch A: Qdrant semantic search
    vec  = embedder.encode(question).tolist()
    hits = qdrant.query_points(collection_name=COLLECTION, query=vec,
                               limit=5, with_payload=True, score_threshold=0.25).points
    semantic_chunks = [h.payload["text"] for h in hits]
    semantic_meta   = [{"title": h.payload["title"], "score": round(h.score, 3),
                        "compound": h.payload["compound"]}
                       for h in hits]

    # Branch B: Neo4j — extract likely entities from question, run traversal
    # Simple heuristic: find compound/disease names that appear in the question
    all_compounds = [r["name"] for r in neo4j_query("MATCH (c:Compound) RETURN c.name AS name")]
    all_diseases  = [r["name"] for r in neo4j_query("MATCH (d:Disease) RETURN d.name AS name")]

    mentioned_compounds = [c for c in all_compounds if c.lower() in question.lower()]
    mentioned_diseases  = [d for d in all_diseases  if d.lower() in question.lower()]

    graph_facts = []

    if mentioned_compounds:
        for comp in mentioned_compounds[:2]:
            rows = neo4j_query("""
                MATCH (c:Compound {name: $name})-[r:TESTED_AGAINST]->(d:Disease)
                MATCH (p:Paper)-[:STUDIES]->(c)
                MATCH (res:Researcher)-[:AUTHORED]->(p)
                RETURN c.name AS compound, d.name AS disease,
                       r.efficacy AS efficacy, res.name AS researcher,
                       p.year AS year, p.method AS method
            """, {"name": comp})
            graph_facts.extend(rows)

    if mentioned_diseases:
        for dis in mentioned_diseases[:2]:
            rows = neo4j_query("""
                MATCH (d:Disease {name: $name})<-[:TARGETS]-(p:Paper)
                MATCH (c:Compound)<-[:STUDIES]-(p)
                RETURN d.name AS disease, c.name AS compound,
                       p.efficacy AS efficacy, p.year AS year
                ORDER BY p.efficacy DESC LIMIT 5
            """, {"name": dis})
            graph_facts.extend(rows)

    # If no entity matched, do a general traversal
    if not graph_facts:
        graph_facts = neo4j_query("""
            MATCH (c:Compound)-[r:TESTED_AGAINST]->(d:Disease)
            MATCH (p:Paper)-[:STUDIES]->(c)
            RETURN c.name AS compound, d.name AS disease,
                   r.efficacy AS efficacy, p.year AS year
            ORDER BY r.efficacy DESC LIMIT 8
        """)

    # Build context
    graph_context = "\n".join([
        f"- {f.get('compound','?')} tested against {f.get('disease','?')}: "
        f"{f.get('efficacy','?')}% efficacy ({f.get('year','?')}) "
        f"by {f.get('researcher', 'unknown')}"
        for f in graph_facts[:8]
    ])

    semantic_context = "\n\n".join(semantic_chunks[:3])

    system_prompt = """You are a medical research assistant with access to a knowledge graph 
of 30 clinical research papers. Answer questions precisely using the provided graph facts and 
semantic evidence. Always cite specific compounds, efficacy percentages, and researchers when available. 
Be concise but complete. Format key facts in bold."""

    user_prompt = f"""Question: {question}

GRAPH FACTS (structured, from Neo4j knowledge graph):
{graph_context if graph_context else "No specific graph facts matched."}

SEMANTIC EVIDENCE (from vector search over paper texts):
{semantic_context[:1200] if semantic_context else "No semantic matches."}

Answer the question using both sources. Mention specific compounds, efficacies, and researchers."""

    response = llm.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=600,
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return {
        "question":        question,
        "answer":          answer,
        "graph_facts":     graph_facts[:8],
        "semantic_sources": semantic_meta,
        "retrieval_info": {
            "entities_matched": {
                "compounds": mentioned_compounds,
                "diseases":  mentioned_diseases,
            },
            "qdrant_hits":  len(hits),
            "graph_facts":  len(graph_facts),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0",
                port=int(os.getenv("BACKEND_PORT", 8000)), reload=True, reload_dirs=["backend"])
