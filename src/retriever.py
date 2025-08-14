import os
import re
import together
import chromadb
import geopandas as gpd
from sentence_transformers import SentenceTransformer
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from together import Together

from config import (
    VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, CHROMA_COLLECTION_NAME
)

# --- Initialize Global Objects ---
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
geolocator = Nominatim(user_agent="geo_rag_app")

# Correctly initialize the Together client with the API key from the environment
try:
    llm_client = Together(api_key="adf223b933e385e6215d78f558f04d9349818eac7b64b4097d64cfdefd120fba")
except Exception as e:
    print(f"Failed to initialize Together client: {e}")
    llm_client = None

def extract_entity_from_query(query: str) -> str:
    """
    Uses the LLM to identify the main landmark or entity in the user's query.
    """
    if not llm_client: return query

    prompt = f"""
From the user question below, extract the full name of the geographical landmark.
Do not add any extra words, explanations, or formatting. Just return the name.

Question: "{query}"

Landmark Name:
    """
    try:
        # UPDATED: Using the 'messages' keyword argument for the chat endpoint
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=50,
            temperature=0.0
        )
        entity = response.choices[0].message.content.strip().replace('"', '')
        print(f"Extracted entity: '{entity}'")
        return entity
    except Exception as e:
        print(f"Could not extract entity: {e}")
        return query

def perform_hybrid_retrieval(query: str, gdf: gpd.GeoDataFrame, top_k: int = 1):
    """
    Performs a geospatial search by geocoding the entity and finding the containing ward.
    """
    entity_name = extract_entity_from_query(query)
    
    if not entity_name or entity_name == query:
        print("Entity extraction failed or returned original query.")
        # Fallback to semantic search if geocoding fails
        # This part can be enhanced later
        return "Could not identify a clear landmark for a precise location search.", gpd.GeoDataFrame()

    try:
        location = geolocator.geocode(f"{entity_name}, Delhi, India")
        if location is None:
            print(f"Could not geocode entity: {entity_name}")
            return "Could not find the location of the specified landmark.", gpd.GeoDataFrame()

        location_point = Point(location.longitude, location.latitude)
        print(f"Geocoded {entity_name} to: {location_point}")

        result_gdf = gdf[gdf.geometry.contains(location_point)]

        if result_gdf.empty:
            return f"Found the location of {entity_name}, but could not match it to a specific ward.", gpd.GeoDataFrame()

    except Exception as e:
        print(f"An error occurred during geospatial search: {e}")
        return "An error occurred while searching for the location.", gpd.GeoDataFrame()


    context_parts = []
    for index, row in result_gdf.iterrows():
        ward_name = row.get('ward_name', 'Unnamed')
        ward_no = row.get('ward_no', 'N/A')
        context_parts.append(f"- The landmark '{entity_name}' is located in ward number {ward_no}, named {ward_name}.")
    
    context_str = "\n".join(context_parts)
    return context_str, result_gdf

def get_llm_response(context: str, query: str) -> str:
    """Generates a response from the LLM based on the retrieved context."""
    if not llm_client: return "LLM client not initialized."

    prompt = f"""
You are an expert on the municipal wards of Delhi. Answer the user's question based ONLY on the context provided below.
The context is highly accurate. State the answer clearly and concisely.

CONTEXT:
---
{context}
---

USER QUESTION: {query}

ANSWER:
    """
    
    try:
        # UPDATED: Using the 'messages' keyword argument for the chat endpoint
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=256,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling Together AI API: {e}")
        return "Sorry, I encountered an error while trying to generate an answer."
