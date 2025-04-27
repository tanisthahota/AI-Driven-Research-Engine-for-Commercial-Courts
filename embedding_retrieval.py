import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# ========== CONFIG ==========
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neo4j123"
CHROMA_DIR = "./chroma_store"
CHROMA_COLLECTION_NAME = "legal_sections"

# ========== CONNECT NEO4J ==========
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ========== CONNECT CHROMADB ==========
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)

# ========== LOAD LEGALBERT ==========
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def embed_text_legalbert(text):
    with torch.no_grad():
        tokens = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        tokens = {k: v.to(device) for k, v in tokens.items()}
        output = model(**tokens)
        embeddings = output.last_hidden_state[:, 0, :]  # CLS token
        return embeddings.cpu().numpy().flatten()

# ========== EXTRACT FROM NEO4J ==========
def get_all_sections(tx):
    query = """
    MATCH (c:Case)-[:HAS_SECTION]->(s:Section)
    RETURN s.id AS section_id, s.label AS label, s.content AS content, 
           c.id AS case_id, c.title AS title, c.type AS type, c.court AS court
    """
    result = tx.run(query)
    return [record.data() for record in result]

with driver.session() as session:
    sections = session.read_transaction(get_all_sections)

print(f"🧠 Retrieved {len(sections)} sections from Neo4j")

# ========== EMBED AND STORE ==========
embeddings = []
metadata = []

for idx, section in enumerate(tqdm(sections, desc="Embedding Sections with LegalBERT")):
    content = section.get("content", "").strip()
    if not content:
        print(f"⚠️ Skipped section {idx}: Empty content")
        continue

    try:
        emb = embed_text_legalbert(content)
        embeddings.append(emb)

        cleaned_meta = {
            "section_id": section.get("section_id", f"unknown_{idx}"),
            "label": section.get("label", ""),
            "case_id": section.get("case_id", ""),
            "title": section.get("title", ""),
            "type": section.get("type", ""),
            "court": section.get("court", ""),
            "content": content
        }
        metadata.append(cleaned_meta)

        #print(f"✅ Embedded Section {idx} | ID: {cleaned_meta['section_id']} | Title: {cleaned_meta['title'][:30]}")
    except Exception as e:
        print(f"❌ Failed to embed Section {idx}: {e}")

print(f"📦 Total embeddings generated: {len(embeddings)}")
print(f"📦 Total metadata entries: {len(metadata)}")

# ========== STORE IN CHROMADB ==========
for i, emb in enumerate(embeddings):
    meta = metadata[i]
    try:
        #print(f"📝 Adding to ChromaDB: Section ID={meta['section_id']} | content={meta['content']}")
        collection.add(
            ids=[str(meta['section_id'])],
            embeddings=[emb.tolist()],
            documents=[meta["content"]],
            metadatas=[meta]
        )
    except Exception as e:
        print(f"❌ Failed to add Section {meta['section_id']} to ChromaDB: {e}")

print("✅ ChromaDB collection populated and persisted.")