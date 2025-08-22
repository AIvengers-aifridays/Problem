import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# ---------- Step 1: Load CSV files from data folder ----------
df1 = pd.read_csv(os.path.join("data", "logs.csv"))   # first CSV
df2 = pd.read_csv(os.path.join("data", "manuals.csv"))  # second CSV

# ---------- Step 2: Normalize schema ----------
df2 = df2.rename(columns={
    "manual_id": "IssueID",
    "component": "System",
    "issue_type": "ErrorType",
    "description": "Description",
    "resolution_steps": "Resolution"
})

# Add missing columns
df2["Date"] = None
df2["RootCause"] = None

# Match df1 column order
df2 = df2[df1.columns]

# Combine both datasets
df_all = pd.concat([df1, df2], ignore_index=True)

# ---------- Step 3: Create text field for embeddings ----------
df_all["text"] = (
    "IssueID: " + df_all["IssueID"].astype(str) +
    " | Date: " + df_all["Date"].astype(str) +
    " | System: " + df_all["System"].astype(str) +
    " | ErrorType: " + df_all["ErrorType"].astype(str) +
    " | Description: " + df_all["Description"].astype(str) +
    " | RootCause: " + df_all["RootCause"].astype(str) +
    " | Resolution: " + df_all["Resolution"].astype(str)
)

# ---------- Step 4: Generate embeddings ----------
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight model
embeddings = model.encode(df_all["text"].tolist(), convert_to_numpy=True)

# ---------- Step 5: Build FAISS index ----------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# ---------- Step 6: Save FAISS index ----------
faiss.write_index(index, os.path.join("data", "vectordb.index"))

# Save metadata separately (to map search results back to rows)
df_all.to_csv(os.path.join("data", "vectordb_metadata.csv"), index=False)

print("✅ VectorDB created: data/vectordb.index and data/vectordb_metadata.csv")
