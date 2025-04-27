import pandas as pd
from neo4j import GraphDatabase

# Load CSV
df = pd.read_csv("filtered_tax_cases.csv")

# Neo4j connection setup (you can customize these)
uri = "bolt://localhost:7687"
username = "neo4j"
password = "neo4j123"

driver = GraphDatabase.driver(uri, auth=(username, password))

# This function creates the graph in Neo4j
def build_graph(tx, df):
    for idx, row in df.iterrows():
        case_id = f"Case_{idx}"
        title = row['Titles'].replace('"', "'")

        # Create main case node
        tx.run(f"""
            MERGE (c:Case {{id: '{case_id}'}})
            SET c.title = "{title}", c.type = "{row['Case_Type']}", c.court = "{row['Court_Name']}"
        """)

        # Create section nodes and connect to case
        for field in ["Facts", "Issues", "PetArg", "RespArg", "Section", "Precedent", "CDiscource", "Conclusion"]:
            content = str(row[field]).strip().replace('"', "'")
            if content and content.lower() != "nan":
                tx.run(f"""
                    MATCH (c:Case {{id: '{case_id}'}})
                    MERGE (s:Section {{id: '{case_id}_{field}'}})
                    SET s.label = "{field}", s.content = "{content}"
                    MERGE (c)-[:HAS_SECTION {{type: '{field}'}}]->(s)
                """)

with driver.session() as session:
    session.write_transaction(build_graph, df)

print("✅ Graph created successfully in Neo4j")