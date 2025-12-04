"""
Script to build a FAISS vector database from the childcare articles CSV.
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

def build_vector_database(csv_path='Child_Care_Articles.csv', 
                         output_dir='vector_db',
                         model_name='all-MiniLM-L6-v2'):  # Fast embedding model
    """
    Build a FAISS vector database from the CSV file.
    
    Args:
        csv_path: Path to the CSV file containing articles
        output_dir: Directory to save the vector database
        model_name: Name of the sentence transformer model
    """
    print("Loading CSV file...")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} articles")
    print("Initializing embedding model...")
    model = SentenceTransformer(model_name)
    
    # Prepare documents
    documents = []
    metadata = []
    
    for idx, row in df.iterrows():
        # Combine category, question, and article for better context
        doc_text = f"Category: {row['Category']}\nQuestion: {row['Question']}\nArticle: {row['Article']}"
        documents.append(doc_text)
        metadata.append({
            'category': row['Category'],
            'question': row['Question'],
            'link': row['Link'],
            'article': row['Article']
        })
    
    print("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True)
    
    # Convert to numpy array
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    print(f"Creating FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    faiss_path = os.path.join(output_dir, 'faiss_index.index')
    faiss.write_index(index, faiss_path)
    print(f"Saved FAISS index to {faiss_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {metadata_path}")
    
    # Save model name for later use
    model_path = os.path.join(output_dir, 'model_name.txt')
    with open(model_path, 'w') as f:
        f.write(model_name)
    print(f"Saved model name to {model_path}")
    
    print(f"\nVector database built successfully!")
    print(f"Total documents: {len(documents)}")
    print(f"Embedding dimension: {dimension}")
    
    return index, metadata, model

if __name__ == "__main__":
    build_vector_database()

