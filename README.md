# movies-vertexai-langchain-neo4j
A gradio chatbot built with movies dataset using integrations of Neo4j, VertexAI and Langchain

## Technology Stack

* **Neo4j**: Graph database for storing comprehensive movie data and vector embeddings of movie plots/overviews
  * Stores movie metadata (title, release date, runtime, ratings)
  * Actor and director information with relationships to movies
  * Genre classifications and other categorical data
  * Production companies and countries
  * Vector embeddings for semantic search capabilities

* **Google Vertex AI**: 
  * Generates vector embeddings from movie plots/overviews
  * Powers semantic search functionality
  * Provides natural language understanding for query processing
  * Enables similarity-based movie recommendations

* **Langchain**: 
  * Orchestrates the connection between components
  * Manages prompt engineering and chain of thought processes
  * Handles retrieval of relevant data from Neo4j

* **Gradio**: 
  * Creates an interactive and user-friendly web interface
  * Provides intuitive UI components for search and filtering
  * Displays search results with visualizations
  * Enables real-time interaction with the underlying models
