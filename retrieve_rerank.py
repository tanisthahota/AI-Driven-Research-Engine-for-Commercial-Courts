import os
import chromadb
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase

# Disable TensorFlow usage
os.environ["USE_TF"] = "0"

CHROMA_DIR = "./chroma_store"
CHROMA_COLLECTION_NAME = "legal_sections"

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

# Initialize Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j123"))

# Load LegalBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

# ----------- Embed Query with LegalBERT -----------
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# ----------- Query ChromaDB -----------
def retrieve_semantic_cases(query, section_type=None, top_k=10):
    query_embedding = embed_text(query)
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    if not results["documents"] or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    candidates = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        case_id = meta.get("case_id", None)
        label = meta.get("label", "").lower()

        if label in section_type or section_type in ["none", "None", "all", "All", "any", "Any"]:
            candidates.append({
                "case_id": case_id,
                "score": 1 / (1 + dist),
                "title": meta.get("title", ""),
                "label": meta.get("label", ""),
                "section_id": meta.get("section_id", ""),
                "court": meta.get("court", ""),
                "type": meta.get("type", ""),
                "content": meta.get("content", ""),
            })

    return candidates

# ----------- Graph Distance via Neo4j -----------
def get_graph_distance(case_id_1, case_id_2, driver, section_type=None):
    with driver.session() as session:
        if section_type:
            result = session.run("""
                MATCH (c1:Case {id: $case1})-[:HAS_SECTION]->(s:Section {title: $section_type})<-[:HAS_SECTION]-(c2:Case {id: $case2})
                RETURN COUNT(s) AS shared_sections
            """, case1=case_id_1, case2=case_id_2, section_type=section_type)
        else:
            result = session.run("""
                MATCH (c1:Case {id: $case1})-[:HAS_SECTION]->(s:Section)<-[:HAS_SECTION]-(c2:Case {id: $case2})
                RETURN COUNT(s) AS shared_sections
            """, case1=case_id_1, case2=case_id_2)

        record = result.single()
        shared_sections = record["shared_sections"] if record else 0
        return 1 / (shared_sections + 1) if shared_sections else float("inf")

# ----------- Reranking Function -----------
def rerank_cases(candidates, query_text, section_type=None, alpha=0.7, beta=0.3):
    reranked_cases = []
    for case in candidates:
        cosine_score = case['score']
        
        # Skip graph scoring if no valid query case ID (e.g., using free-form query)
        try:
            graph_score = get_graph_distance(case['case_id'], query_text, driver, section_type)
            if graph_score == float("inf"):
                final_score = cosine_score
            else:
                final_score = alpha * cosine_score + beta * (1 / (1 + graph_score))
        except:
            final_score = cosine_score

        reranked_cases.append({
            "case_id": case["case_id"],
            "score": final_score,
            "title": case.get("title", ""),
            "label": case.get("label", ""),
            "section_id": case.get("section_id", ""),
            "court": case.get("court", ""),
            "type": case.get("type", ""),
            "content": case.get("content", ""),
        })

    return sorted(reranked_cases, key=lambda x: x['score'], reverse=True)

# ----------- Main Pipeline -----------
def main(query_text, section_type=None):
    print(f"\n[INFO] Running query: '{query_text}'")

    candidates = retrieve_semantic_cases(query_text, section_type=section_type)
    if not candidates:
        print("[WARN] No candidates found.")
        return []

    # Use query text as case ID placeholder (assuming this is an actual ID — replace with real ID if needed)
    for candidate in candidates:
        candidate["query_case_id"] = query_text  # Just a placeholder; ideally this would be a proper case ID

    reranked = rerank_cases(candidates, query_text, section_type=section_type)
    return reranked

# ----------- Example Run -----------
if __name__ == "__main__":
    query = "latest ruling on economic duress"
    section_type = "any"  # Optional: "Issue", "Precedent Analysis", etc.

    reranked_results = main(query_text=query, section_type=section_type)

    for result in reranked_results:
        print(f"[{result['score']:.4f}] [Case ID:{result['case_id']}] {result['title']} ({result['label']})\n→ {result['content']}\n")