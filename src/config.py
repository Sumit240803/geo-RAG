import os

# --- Directory Paths ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
VECTOR_STORE_DIR = os.path.join(ROOT_DIR, "vector_store")

# --- Data File Paths ---
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "delhi_wards.geojson")

# UPDATED: Path to the processed GeoDataFrame using the Parquet format
GDF_PICKLE_PATH = os.path.join(DATA_DIR, "processed", "delhi_wards_gdf.parquet")

# --- Model Configurations ---
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = "meta-llama/Llama-3-8b-chat-hf"

# --- ChromaDB Configuration ---
CHROMA_COLLECTION_NAME = "delhi_wards"
