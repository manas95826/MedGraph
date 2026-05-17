"""
backend/ingest.py
─────────────────
Ingestion pipeline:
  1. Cognee LLM extraction → writes enriched entities to Neo4j + Qdrant (cognee collections)
  2. Direct Qdrant → writes medgraph_chunks collection for semantic search (after cognee prune)
  3. Direct Neo4j  → writes typed schema nodes + relationships (after cognee prune)

Run: python backend/ingest.py
"""

import os, json, asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import cognee
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

NEO4J_URI   = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER  = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS  = os.getenv("NEO4J_PASSWORD", "medgraph123")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6335))
COLLECTION  = os.getenv("QDRANT_COLLECTION", "medgraph_chunks")

EMBED_DIM = 384  # all-MiniLM-L6-v2


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cognee — LLM entity extraction → Qdrant (cognee collections) + Neo4j
# ─────────────────────────────────────────────────────────────────────────────
def configure_cognee():
    os.environ.setdefault("ENV", "dev")  # QDrantAdapter does os.getenv("ENV").lower()
    cognee.config.set_llm_config({
        "llm_provider": "openai",
        "llm_model":    os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "llm_api_key":  os.getenv("OPENAI_API_KEY"),
    })
    # Qdrant as cognee's vector store — port included in URL so adapter uses it
    cognee.config.set_vector_db_config({
        "vector_db_provider": "qdrant",
        "vector_db_url":      f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        "vector_db_key":      "local",  # adapter requires non-empty; no real key for local Qdrant
    })
    cognee.config.set_graph_db_config({
        "graph_database_provider": "neo4j",
        "graph_database_url":      NEO4J_URI,
        "graph_database_username": NEO4J_USER,
        "graph_database_password": NEO4J_PASS,
    })
    print("✓ Cognee configured (vector=qdrant, graph=neo4j)")


async def ingest_cognee(papers: list[dict]):
    """
    Feeds paper texts into Cognee:
      - prune_system wipes Qdrant + Neo4j for a clean slate
      - cognify: chunk → GPT-4o-mini entity extraction → embed → write to Qdrant + Neo4j
    Note: prune_system runs first, so direct Qdrant + Neo4j ingestion must run AFTER this.
    """
    await cognee.prune.prune_system(metadata=True)

    texts = [p["text"] for p in papers]
    await cognee.add(texts, dataset_name="medical_papers")
    print("  Cognee: texts added, running cognify (LLM extraction)...")

    await cognee.cognify()
    print("✓ Cognee: knowledge graph construction complete")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Qdrant — medgraph_chunks collection for the API's semantic search
# ─────────────────────────────────────────────────────────────────────────────
def ingest_qdrant(papers: list[dict]):
    """
    Writes one vector per paper into the medgraph_chunks collection.
    Runs after ingest_cognee so prune_system doesn't wipe it.
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    model  = SentenceTransformer("all-MiniLM-L6-v2")

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    points = []
    for i, p in enumerate(papers):
        vec = model.encode(p["text"]).tolist()
        points.append(PointStruct(
            id=i,
            vector=vec,
            payload={
                "paper_id":    p["id"],
                "title":       p["title"],
                "year":        p["year"],
                "compound":    p["compound"],
                "disease":     p["disease"],
                "institution": p["institution"],
                "journal":     p["journal"],
                "method":      p["method"],
                "efficacy":    p["efficacy_pct"],
                "text":        p["text"],
            }
        ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"✓ Qdrant: {len(points)} vectors uploaded to '{COLLECTION}'")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Neo4j — typed nodes + relationships for precise Cypher queries
# ─────────────────────────────────────────────────────────────────────────────
def ingest_neo4j(papers: list[dict]):
    """
    Writes Paper, Researcher, Institution, Compound, Disease, Journal nodes
    and all typed edges. Runs after ingest_cognee so prune_system doesn't wipe it.
    Uses MERGE so re-runs are idempotent.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    with driver.session() as session:
        # Clear cognee's internal __Node__ schema, keep only our typed nodes
        session.run("MATCH (n) WHERE NOT n:Paper AND NOT n:Researcher AND NOT n:Institution "
                    "AND NOT n:Compound AND NOT n:Disease AND NOT n:Journal DETACH DELETE n")

        # Uniqueness constraints
        for label in ["Paper", "Researcher", "Institution", "Compound", "Disease", "Journal"]:
            prop = "id" if label == "Paper" else "name"
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE")

        for p in papers:
            session.run("""
                MERGE (paper:Paper {id: $id})
                SET paper.title    = $title,
                    paper.year     = $year,
                    paper.method   = $method,
                    paper.efficacy = $efficacy,
                    paper.text     = $text
            """, id=p["id"], title=p["title"], year=p["year"],
                 method=p["method"], efficacy=p["efficacy_pct"], text=p["text"])

            session.run("""
                MERGE (r:Researcher {name: $name})
                MERGE (paper:Paper {id: $pid})
                MERGE (r)-[:AUTHORED]->(paper)
            """, name=p["lead_author"], pid=p["id"])

            session.run("""
                MERGE (i:Institution {name: $name})
                MERGE (r:Researcher {name: $rname})
                MERGE (r)-[:AFFILIATED_WITH]->(i)
                MERGE (paper:Paper {id: $pid})
                MERGE (paper)-[:CONDUCTED_AT]->(i)
            """, name=p["institution"], rname=p["lead_author"], pid=p["id"])

            session.run("""
                MERGE (ci:Institution {name: $name})
                MERGE (paper:Paper {id: $pid})
                MERGE (paper)-[:COLLABORATED_WITH]->(ci)
            """, name=p["collaborating_institution"], pid=p["id"])

            session.run("""
                MERGE (c:Compound {name: $name})
                SET c.type = $ctype
                MERGE (paper:Paper {id: $pid})
                MERGE (paper)-[:STUDIES]->(c)
            """, name=p["compound"], ctype=p["compound_type"], pid=p["id"])

            session.run("""
                MERGE (d:Disease {name: $name})
                MERGE (paper:Paper {id: $pid})
                MERGE (paper)-[:TARGETS]->(d)
                MERGE (c:Compound {name: $cname})
                MERGE (c)-[:TESTED_AGAINST {efficacy: $eff}]->(d)
            """, name=p["disease"], pid=p["id"],
                 cname=p["compound"], eff=p["efficacy_pct"])

            session.run("""
                MERGE (j:Journal {name: $name})
                MERGE (paper:Paper {id: $pid})
                MERGE (paper)-[:PUBLISHED_IN]->(j)
            """, name=p["journal"], pid=p["id"])

    driver.close()
    print(f"✓ Neo4j: {len(papers)} papers + full typed entity graph written")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    papers_path = Path("data/papers.json")
    if not papers_path.exists():
        raise FileNotFoundError("Run: python scripts/generate_data.py first")

    with open(papers_path) as f:
        papers = json.load(f)

    print(f"\n{'='*50}")
    print(f"  MedGraph Ingestion Pipeline")
    print(f"  Papers: {len(papers)}")
    print(f"{'='*50}\n")

    configure_cognee()

    print("\n[1/3] Running Cognee LLM extraction pipeline...")
    print("  (5-10 mins — GPT-4o-mini processes each paper)")
    asyncio.run(ingest_cognee(papers))

    print("\n[2/3] Writing medgraph_chunks → Qdrant...")
    ingest_qdrant(papers)

    print("\n[3/3] Writing typed knowledge graph → Neo4j...")
    ingest_neo4j(papers)

    print("\n✅ All done! Run: cd backend && python api.py")
