from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from datetime import datetime
import re

from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MovieChunkConfig:
    """Configuration for chunk creation and storage"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_directory: str = "./chroma_db"
    collection_name: str = "movie_recommendations"

class ConversationAnalyzer:
    """Analyzes movie conversations for relevant patterns"""
    
    GENRE_PATTERNS = {
        'action': r'\b(action|thriller|adventure)\b',
        'comedy': r'\b(comedy|funny|humorous|hilarious)\b',
        'drama': r'\b(drama|dramatic|emotional)\b',
        'romance': r'\b(romance|romantic|love story)\b',
        'family': r'\b(family|children|kids)\b',
        'musical': r'\b(musical|music|singing|dance)\b'
    }
    
    MOOD_PATTERNS = {
        'exciting': r'\b(exciting|thrilling|intense)\b',
        'heartwarming': r'\b(heartwarming|touching|sweet)\b',
        'funny': r'\b(funny|hilarious|amusing)\b',
        'serious': r'\b(serious|thought-provoking|deep)\b'
    }
    
    @classmethod
    def analyze_text(cls, text: str) -> Dict[str, str]:
        """Analyze text and return ChromaDB-compatible metadata"""
        text = text.lower()
        
        # Extract genres
        genres = []
        for genre, pattern in cls.GENRE_PATTERNS.items():
            if re.search(pattern, text):
                genres.append(genre)
        
        # Extract moods
        moods = []
        for mood, pattern in cls.MOOD_PATTERNS.items():
            if re.search(pattern, text):
                moods.append(mood)
        
        return {
            "genres": ",".join(genres) if genres else "none",
            "moods": ",".join(moods) if moods else "none",
            "genre_count": str(len(genres)),
            "mood_count": str(len(moods))
        }

class MovieChunkManager:
    """Manages movie context chunks and vector storage"""
    
    def __init__(self, config: MovieChunkConfig = MovieChunkConfig()):
        self.config = config
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load existing vector store"""
        return Chroma(
            persist_directory=self.config.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name
        )
    
    def process_conversation(
        self,
        user_data: Dict[str, Any],
        conversation: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Process conversation into text and metadata"""
        
        # Extract basic data
        user_id = user_data['user']['amazon_id']
        conv_text = conversation.get('text', '')
        conv_id = str(conversation.get('conversation_id', ''))
        
        # Get movie interactions
        liked_movies = [m['title'] for m in conversation.get('movies', {}).get('liked', [])]
        disliked_movies = [m['title'] for m in conversation.get('movies', {}).get('disliked', [])]
        recommended_movies = [m['title'] for m in conversation.get('movies', {}).get('recommended', [])]
        
        # Analyze conversation
        analysis = ConversationAnalyzer.analyze_text(conv_text)
        
        # Create chunk text
        chunk_text = f"""
        User ID: {user_id}
        Conversation ID: {conv_id}
        
        Movie Preferences:
        Liked: {', '.join(liked_movies) if liked_movies else 'None'}
        Disliked: {', '.join(disliked_movies) if disliked_movies else 'None'}
        Recommended: {', '.join(recommended_movies) if recommended_movies else 'None'}
        
        Conversation:
        {conv_text}
        """
        
        # Create metadata
        metadata = {
            "user_id": user_id,
            "conversation_id": conv_id,
            "timestamp": datetime.now().isoformat(),
            "liked_movies": ",".join(liked_movies) if liked_movies else "none",
            "disliked_movies": ",".join(disliked_movies) if disliked_movies else "none",
            "recommended_movies": ",".join(recommended_movies) if recommended_movies else "none",
            "like_count": str(len(liked_movies)),
            "dislike_count": str(len(disliked_movies)),
            **analysis
        }
        
        return chunk_text, metadata
    
    def process_dataset(
        self,
        dataset_path: Path,
        batch_size: int = 10
    ) -> None:
        """Process movie dataset and store in vector DB"""
        try:
            logger.info(f"Processing dataset from {dataset_path}")
            docs_to_store = []
            
            with open(dataset_path, 'r') as f:
                for line in f:
                    user_data = json.loads(line)
                    
                    # Process each conversation
                    for conversation in user_data.get('conversations', []):
                        # Create text and metadata
                        text, metadata = self.process_conversation(user_data, conversation)
                        
                        # Create Document object
                        doc = Document(
                            page_content=text,
                            metadata=metadata
                        )
                        docs_to_store.append(doc)
                        
                        # Process in batches
                        if len(docs_to_store) >= batch_size:
                            self._store_documents(docs_to_store)
                            docs_to_store = []
            
            # Store remaining documents
            if docs_to_store:
                self._store_documents(docs_to_store)
            
            logger.info("Dataset processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise
    
    def _store_documents(self, documents: List[Document]) -> None:
        """Store documents in vector store"""
        try:
            # Directly store documents
            self.vector_store.add_documents(documents)
            logger.info(f"Stored {len(documents)} chunks in vector store")
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            raise
    
    def get_similar_chunks(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        k: int = 3
    ) -> List[Document]:
        """Retrieve similar chunks with filtering"""
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )

def main():
    """Example usage of the MovieChunkManager"""
    try:
        # Initialize manager
        config = MovieChunkConfig()
        manager = MovieChunkManager(config)
        
        # Process dataset
        dataset_path = Path("LLM_Redial/Movie/fully_mapped_user_ids_and_movie_ids.jsonl")
        manager.process_dataset(dataset_path)
        
        # Test retrieval
        similar_chunks = manager.get_similar_chunks(
            "recommend a family movie with great performances",
            filter_dict={"genres": "family"}
        )
        
        logger.info(f"Found {len(similar_chunks)} relevant chunks")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()