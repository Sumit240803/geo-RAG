# --- ChromaDB/SQLite3 Workaround for Streamlit Cloud ---
# This code block must be at the very top of this file.
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End of Workaround ---

import streamlit as st
import geopandas as gpd
import os
from dotenv import load_dotenv
import pydeck as pdk

from config import GDF_PICKLE_PATH, VECTOR_STORE_DIR
# UPDATED: Import the retriever module itself, not its functions directly
import retriever
# UPDATED: Import the main function from data_processing
from data_processing import main as run_data_processing

# --- Page Configuration ---
st.set_page_config(
    page_title="Geo-RAG: Delhi Wards",
    page_icon="🗺️",
    layout="wide"
)

# --- Load Environment Variables ---
load_dotenv() 

# --- Self-Initializing Data Setup ---
# This block ensures that the data processing runs automatically if the DB is missing.
if not os.path.exists(VECTOR_STORE_DIR) or not os.path.exists(GDF_PICKLE_PATH):
    st.info("First-time setup: The data is being processed. This may take a few minutes...")
    with st.spinner("Downloading data, creating embeddings, and building the database..."):
        run_data_processing()
    st.success("Data processing complete! The app is ready.")
    st.rerun()

# --- Caching Data and Initializing Retriever ---
@st.cache_resource
def initialize_retriever():
    """
    Initializes all the necessary components for the retriever.
    Using @st.cache_resource ensures this is done only once.
    """
    return retriever.GeoRetriever()

@st.cache_data
def load_geodata():
    """
    Loads the preprocessed GeoDataFrame from a Parquet file.
    """
    if not os.path.exists(GDF_PICKLE_PATH):
        st.error(f"Processed data file not found at {GDF_PICKLE_PATH}. Please run `src/data_processing.py` first.")
        return None
    return gpd.read_parquet(GDF_PICKLE_PATH)

# --- Pydeck Map Function ---
def create_map(data, zoom):
    """Creates a pydeck map that can render polygons."""
    center_lat = data.geometry.centroid.y.mean()
    center_lon = data.geometry.centroid.x.mean()

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
    )

    layer = pdk.Layer(
        "GeoJsonLayer",
        data,
        opacity=0.6,
        stroked=True,
        filled=True,
        extruded=False,
        wireframe=True,
        get_fill_color="[255, 0, 0, 140]",
        get_line_color=[255, 255, 255],
        pickable=True,
        auto_highlight=True
    )
    
    tooltip = {
        "html": "<b>Ward Name:</b> {ward_name}<br/><b>Ward No:</b> {ward_no}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }

    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(deck)

# --- Main Application UI ---
def main():
    st.title("🗺️ Geographic RAG for Delhi Wards")
    st.markdown("Ask a question about the municipal wards of Delhi.")
    
    # Initialize the retriever and load data
    geo_retriever = initialize_retriever()
    gdf = load_geodata()
    
    if gdf is None: 
        return

    with st.sidebar:
        st.header("Query")
        query_text = st.text_input("Enter your question:", "Which ward is the Lotus Temple in?")
        search_button = st.button("Search", type="primary")
        st.info("This demo uses a dataset of Delhi's 272 municipal wards.")

    if search_button and query_text:
        if not os.environ.get("TOGETHER_API_KEY"):
            st.error("TOGETHER_API_KEY not found. Please add it to your .env file.")
            return

        with st.spinner("Searching wards and asking the LLM..."):
            # UPDATED: Call methods from the retriever object
            context_str, result_gdf = geo_retriever.perform_hybrid_retrieval(query_text, gdf)
            if result_gdf.empty:
                st.warning("Could not find a relevant ward. Please try another question.")
                return

            llm_answer = geo_retriever.get_llm_response(context=context_str, query=query_text)
            st.subheader("Answer")
            st.markdown(llm_answer)
            st.subheader("Found Wards on Map")
            
            create_map(result_gdf, zoom=12)
            
            with st.expander("Show Retrieved Context"):
                st.text(context_str)
            with st.expander("Show Data View of Results"):
                st.dataframe(result_gdf[['ward_name', 'ward_no']])
    else:
        st.info("Enter a query on the left to begin.")
        create_map(gdf, zoom=9)

if __name__ == "__main__":
    main()
