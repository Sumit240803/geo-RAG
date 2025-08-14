# Geo-RAG: An Intelligent Geospatial Question-Answering System

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-green)](https://streamlit.io/)


An advanced **Geospatial Retrieval-Augmented Generation (Geo-RAG)** system to answer complex, location-based questions about Delhi using LLMs enhanced with geospatial retrieval.

---

## 📌 Table of Contents

1. [Features](#-features)
2. [System Architecture](#-system-architecture)
3. [Tech Stack](#-tech-stack)
4. [Setup and Installation](#-setup-and-installation)
5. [Running the Application](#-running-the-application)

---

## Features

* **Natural Language Queries:** Ask questions in plain English (e.g., "Which ward is the Red Fort in?").
* **Geospatial Retrieval:** Accurately identifies landmarks, finds their geographic coordinates, and determines which administrative ward contains them.
* **Interactive Map:** Visualizes retrieved ward boundaries on an interactive map.
* **Modular Architecture:** Easy to understand, maintain, and adapt for other cities or datasets.

---

## System Architecture

1. **Data Ingestion (Offline):**

   * Downloads GeoJSON data for Delhi's municipal wards.
   * Generates descriptive embeddings.
   * Stores them in a local ChromaDB vector store.

2. **User Query (Online):**

   * Enter a query into the Streamlit web app.

3. **Entity Extraction:**

   * LLM identifies the main landmark in the query (e.g., "Red Fort").

4. **Geocoding:**

   * Gets precise latitude and longitude for the extracted landmark.

5. **Spatial Query:**

   * Finds the exact ward polygon that contains the landmark's coordinates.

6. **Response Generation:**

   * Retrieves context is passed to the LLM to generate a final, human-readable answer.

---

## Tech Stack

* **Backend:** Python 3.10+
* **Web Framework:** Streamlit
* **Geospatial:** GeoPandas, Pydeck, Geopy, Shapely
* **Vector Database:** ChromaDB
* **LLM & Embeddings:** Together AI, Sentence-Transformers

---

##  Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd geo-rag-app
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add your Together AI API key:

```env
TOGETHER_API_KEY="your_api_key_here"
```

---

##  Running the Application

### Step 1: Process the Data (One-Time Setup)

```bash
python src/data_processing.py
```

### Step 2: Run the Streamlit App

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the app.

---


