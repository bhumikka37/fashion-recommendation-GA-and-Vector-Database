import weaviate
from weaviate.classes.init import Auth
from sentence_transformers import SentenceTransformer
import pandas as pd
client = weaviate.connect_to_wcs(
    cluster_url="https://vjdlfn7rbum0pszjhpt9g.c0.asia-southeast1.gcp.weaviate.cloud",
    auth_credentials=Auth.api_key("MZHQuXA8XdafltJfaTI0plJV1AZFz1JBqrTN")
)

# Generate a Python script to upload fashion product data with embeddings into a Weaviate instance

df = pd.read_csv("C:\\Users\\Bhumikka Pancharane\\OneDrive\\Desktop\\Sem IV Notes and QB\\Datasets\\Fashion_Dataset.csv")
df.dropna(subset=['p_id', 'name', 'img', 'description', 'price', 'avg_rating'], inplace=True)

# Define schema if not exists
schema = {
    "class": "Product",
    "description": "A class representing fashion products",
    "vectorizer": "none",  # We provide our own vectors
    "properties": [
        {"name": "p_id", "dataType": ["string"]},
        {"name": "name", "dataType": ["text"]},
        {"name": "description", "dataType": ["text"]},
        {"name": "img", "dataType": ["text"]},
        {"name": "price", "dataType": ["number"]},
        {"name": "avg_rating", "dataType": ["number"]}
    ]
}
collection = client.collections.get("Product")

# Embed descriptions
embedder = SentenceTransformer("all-MiniLM-L6-v2")
description_embeddings = embedder.encode(df['description'].tolist(), show_progress_bar=True)

# Upload data
for i, (idx,row) in enumerate(df.iterrows()):
    properties = {
        "p_id": str(row["p_id"]),
        "name": row["name"],
        "description": row["description"],
        "img": row["img"],
        "price": float(row["price"]),
        "avg_rating": float(row["avg_rating"])
    }

    try:
        collection.data.insert(properties=properties,vector=description_embeddings[i].tolist())
        print(f"Uploaded row{i}")
    except Exception as e:
        print(f"Failed to upload row {i}: {e}")
