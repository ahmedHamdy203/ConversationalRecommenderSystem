# Movie Recommendation System Documentation

## Overview

This project implements a movie recommendation system using a combination of Retrieval-Augmented Generation (RAG) and pure few-shot learning approaches. The system is designed to provide personalized movie recommendations to users based on their stated preferences and conversation history.

The system consists of the following main components:

1. **Data Loading and Preprocessing**
2. **Movie Chunk Management and Retrieval**
3. **Few-shot Learning Recommender**
4. **RAG-based Recommender**
5. **REST API Integration**
6. **Testing Scripts**

Let's dive into the details of each component.

## 1. Data Loading and Preprocessing (data_loader.py)

This module is responsible for loading and processing the movie dataset. It performs the following tasks:

- Reads data from various JSON and text files, including user data, movie information, and conversation logs.
- Maps movie IDs to their corresponding titles.
- Creates a fully mapped dataset that can be used by the recommendation system.
- Provides methods to export the processed dataset to a JSONL file.

Key features:
- Handles missing or inconsistent data gracefully.
- Supports efficient processing of large datasets.
- Ensures data consistency and integrity.

## 2. Movie Chunk Management and Retrieval (movie_chunk_manager.py)

This module manages the storage and retrieval of movie conversation chunks using a vector store (Chroma). It performs the following tasks:

- Processes conversations, extracting relevant metadata (genres, moods, etc.).
- Stores the conversation chunks in the vector store.
- Provides a method to retrieve similar chunks based on a given query and filtering.

Key features:
- Supports efficient storage and retrieval of conversation chunks.
- Leverages the Chroma vector store for fast similarity search.
- Includes metadata extraction and filtering capabilities.

## 3. Few-shot Learning Recommender (movie_few_shot_learning.py)

This module implements a pure few-shot learning-based movie recommender, without using any retrieval. It performs the following tasks:

- Defines a set of comprehensive examples covering different movie genres and user preferences.
- Uses these examples to generate recommendations based on the user's stated preferences.

Key features:
- Provides a simple and straightforward recommendation approach.
- Can be used as a fallback or complementary method to the RAG-based recommender.
- Demonstrates the effectiveness of few-shot learning for movie recommendations.

## 4. RAG-based Recommender (movie_rag_pipeline.py)

This module implements the full Retrieval-Augmented Generation (RAG) pipeline for movie recommendations. It performs the following tasks:

- Downloads and loads the necessary models (LLM and embeddings).
- Sets up the conversation memory using a ConversationBufferWindowMemory.
- Creates the RAG chain that combines retrieval and generation components.
- Provides methods to generate responses, retrieve chat history, and clear conversation history.

Key features:
- Leverages the power of the RAG approach for personalized recommendations.
- Efficiently manages conversation history and user preferences.
- Integrates with the Chroma vector store for retrieval.

## 5. REST API Integration (movie_rag_pipeline.py)

The movie recommendation system is exposed as a REST API using FastAPI. The API provides the following endpoints:

- `/recommend/`: Generates a movie recommendation based on the user's input and context.
- `/history/`: Retrieves the conversation history.
- `/history/`: Clears the conversation history.

Key features:
- Allows external applications to interact with the movie recommendation system.
- Provides a standardized interface for accessing the system's functionality.
- Supports different HTTP methods (POST, GET, DELETE) for the various endpoints.

## 6. Testing Scripts (testing_REST_api_fewshot.py, testing_REST_api_rag.py)

The project includes two testing scripts to validate the functionality of the movie recommendation system's REST API:

1. **testing_REST_api_fewshot.py**:
   - Tests the movie recommendation API that uses the few-shot learning approach.
   - Defines a set of test cases and checks the API's responses.
   - Saves the test results to a JSON file for later review.

2. **testing_REST_api_rag.py**:
   - Tests the movie recommendation API that uses the RAG approach.
   - Defines a set of test queries and user context information.
   - Checks the responses from the `/recommend/`, `/history/`, and `/history/` (clear) endpoints.

Key features:
- Ensures the reliability and correctness of the movie recommendation system's API.
- Provides a framework for regression testing and continuous integration.
- Allows easy comparison of test results over time.