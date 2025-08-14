import geopandas as gpd
from sentence_transformers import SentenceTransformer
import chromadb
import os
import shutil
import requests

from config import (
    RAW_DATA_PATH, GDF_PICKLE_PATH, VECTOR_STORE_DIR,
    EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME
)

def download_data_if_needed():
    """Checks if the data file exists, and downloads it if it doesn't."""
    data_url = "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Delhi/Delhi_Wards.geojson"
    
    raw_data_dir = os.path.dirname(RAW_DATA_PATH)
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)
        print(f"Created directory: {raw_data_dir}")

    if not os.path.exists(RAW_DATA_PATH):
        print(f"Data file not found. Downloading from {data_url}...")
        try:
            response = requests.get(data_url, timeout=30)
            response.raise_for_status()
            with open(RAW_DATA_PATH, 'wb') as f:
                f.write(response.content)
            print("Download successful!")
        except requests.exceptions.RequestException as e:
            print(f"FATAL ERROR: Could not download data file. Error: {e}")
            return False
    else:
        print("Data file already exists. Skipping download.")
    return True

def create_feature_document(row):
    """Creates a descriptive text document for a municipal ward."""
    # This function now reliably uses lowercase column names
    ward_name = row.get('ward_name', 'Unnamed Ward')
    ward_no = row.get('ward_no', 'N/A')
    return f"This is municipal ward number {ward_no}, named {ward_name}, in Delhi."

def main():
    print("--- Starting Data Processing for Delhi Wards ---")

    if not download_data_if_needed():
        return

    gdf = gpd.read_file(RAW_DATA_PATH)
    
    # UPDATED: A more robust way to handle column names.
    # This converts all column names to lowercase, regardless of original capitalization.
    gdf.columns = gdf.columns.str.lower()
    
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(gdf)} ward features.")

    gdf['document'] = gdf.apply(create_feature_document, axis=1)

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Creating embeddings for documents...")
    embeddings = embedding_model.encode(gdf['document'].tolist(), show_progress_bar=True)

    print("Setting up ChromaDB vector store...")
    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)
    
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    print("Adding embeddings to ChromaDB...")
    ids = gdf.index.astype(str).tolist()
    collection.add(ids=ids, embeddings=embeddings.tolist(), documents=gdf['document'].tolist())
    print(f"Successfully added {collection.count()} items.")

    print("Saving processed GeoDataFrame...")
    os.makedirs(os.path.dirname(GDF_PICKLE_PATH), exist_ok=True)
    gdf.drop(columns=['document'], inplace=True)
    gdf.to_parquet(GDF_PICKLE_PATH)
    
    print("--- Data Processing Complete ---")

if __name__ == "__main__":
    main()
