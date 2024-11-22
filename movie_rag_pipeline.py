
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import os

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RagConfig:
    """Configuration for RAG pipeline"""
    # Model settings
    model_repo_id: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    model_filename: str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    model_type: str = "llama"
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # Retriever settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_directory: str = "./chroma_db"
    collection_name: str = "movie_recommendations"
    retriever_k: int = 3
    
    # Memory settings
    max_history_length: int = 5
    
    def __post_init__(self):
        """Ensure model directory exists"""
        self.model_dir = Path("./models")
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / self.model_filename

class ModelManager:
    """Handles model downloading and loading"""
    
    @staticmethod
    def download_model(config: RagConfig) -> Path:
        """Download model if not exists"""
        try:
            if not config.model_path.exists():
                logger.info(f"Downloading model from {config.model_repo_id}")
                hf_hub_download(
                    repo_id=config.model_repo_id,
                    filename=config.model_filename,
                    local_dir=config.model_dir
                )
                logger.info("Model downloaded successfully")
            else:
                logger.info("Using existing model file")
            
            return config.model_path
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise

class MovieRecommenderRAG:
    """RAG pipeline for movie recommendations with conversation management"""
    
    def __init__(self, config: RagConfig = RagConfig()):
        self.config = config
        
        # Download model if needed
        self.model_path = ModelManager.download_model(config)
        
        # Initialize components
        self.memory = self._setup_memory()
        self.llm = self._setup_llm()
        self.embeddings = self._setup_embeddings()
        self.vector_store = self._setup_vector_store()
        self.retriever = self._setup_retriever()
        
        # Create the chain
        self.chain = self._create_chain()
    
    def _setup_memory(self) -> ConversationBufferWindowMemory:
        """Initialize conversation memory with window"""
        return ConversationBufferWindowMemory(
            k=self.config.max_history_length,
            memory_key="chat_history",
            output_key="response",
            return_messages=True
        )
    
    def _setup_llm(self) -> CTransformers:
        """Initialize the LLM"""
        return CTransformers(
            model=str(self.model_path),
            model_type=self.config.model_type,
            config={
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "context_length": 2048,
                "gpu_layers": 0
            }
        )
    
    def _setup_embeddings(self) -> SentenceTransformerEmbeddings:
        """Initialize embeddings"""
        return SentenceTransformerEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
    
    def _setup_vector_store(self) -> Chroma:
        """Initialize vector store"""
        return Chroma(
            persist_directory=self.config.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name
        )
    
    def _setup_retriever(self):
        """Initialize retriever with MMR"""
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.config.retriever_k,
                "fetch_k": self.config.retriever_k * 2
            }
        )
    
    def _create_chain(self):
        """Create the RAG chain with conversation memory"""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a friendly and knowledgeable movie recommendation assistant. 
            Use the provided conversation context and movie information to make personalized recommendations.
            Consider the user's explicit preferences and past interactions when making suggestions.
            If you're not sure about something, acknowledge the uncertainty and focus on what you do know.
            Always provide brief explanations for your recommendations."""),
            
            MessagesPlaceholder(variable_name="chat_history"),
            
            HumanMessage(content="""User Query: {query}
            
            Relevant Movie Information:
            {context}
            
            Previous User Preferences:
            {user_preferences}
            
            Please provide a thoughtful recommendation based on this information.""")
        ])
        
        # Format documents
        def format_docs(docs):
            return "\n".join(f"Movie Context {i+1}:\n{doc.page_content}" 
                           for i, doc in enumerate(docs))
        
        # Extract user preferences
        def get_user_preferences(chat_history):
            preferences = {
                "liked_movies": set(),
                "disliked_movies": set(),
                "genres": set(),
                "moods": set()
            }
            
            for message in chat_history:
                text = message.content.lower()
                # Simple preference extraction
                if "like" in text or "enjoy" in text:
                    # Add positive preferences
                    preferences["liked_movies"].update(
                        movie for movie in text.split('"') 
                        if movie.strip() and len(movie) > 4
                    )
                if "dislike" in text or "hate" in text:
                    # Add negative preferences
                    preferences["disliked_movies"].update(
                        movie for movie in text.split('"')
                        if movie.strip() and len(movie) > 4
                    )
            
            return "\n".join([
                f"Liked Movies: {', '.join(preferences['liked_movies'])}",
                f"Disliked Movies: {', '.join(preferences['disliked_movies'])}",
                f"Preferred Genres: {', '.join(preferences['genres'])}",
                f"Preferred Moods: {', '.join(preferences['moods'])}"
            ])
        
        # Create the chain
        chain = (
            {
                "query": RunnablePassthrough(),
                "chat_history": lambda _: self.memory.load_memory_variables({})["chat_history"],
                "context": lambda x: format_docs(self.retriever.get_relevant_documents(x["query"])),
                "user_preferences": lambda x: get_user_preferences(
                    self.memory.load_memory_variables({})["chat_history"]
                )
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    async def generate_response(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response based on query and context"""
        try:
            # Log the interaction
            logger.info(f"Processing query: {query}")
            
            # Get response from chain
            response = await self.chain.ainvoke({
                "query": query,
                "user_context": user_context
            })
            
            # Update memory
            self.memory.save_context(
                {"input": query},
                {"response": response}
            )
            
            # Get updated history
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            
            result = {
                "response": response,
                "chat_history": chat_history,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Successfully generated response")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get formatted chat history"""
        try:
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            return [
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": msg.content
                }
                for i, msg in enumerate(chat_history)
            ]
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            raise
    
    def clear_history(self):
        """Clear conversation history"""
        self.memory.clear()
        logger.info("Conversation history cleared")

# FastAPI integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    user_context: Optional[Dict[str, Any]] = None

class MovieRecommenderAPI:
    def __init__(self):
        self.rag = MovieRecommenderRAG()

    async def get_recommendation(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None
    ):
        try:
            return await self.rag.generate_response(query, user_context)
        except Exception as e:
            logger.error(f"API Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize API handler
recommender = MovieRecommenderAPI()

@app.post("/recommend/")
async def get_recommendation(request: QueryRequest):
    """Get movie recommendation"""
    return await recommender.get_recommendation(request.query, request.user_context)

@app.get("/history/")
async def get_history():
    """Get conversation history"""
    return {"history": recommender.rag.get_chat_history()}

@app.delete("/history/")
async def clear_history():
    """Clear conversation history"""
    recommender.rag.clear_history()
    return {"message": "Chat history cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)