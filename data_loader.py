from typing import Dict, List, Optional, Any
import json
from dataclasses import dataclass
from pathlib import Path
import logging
import sys
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation with mapped movie titles."""
    conversation_id: int
    user_likes: List[str]  # Movie titles
    user_dislikes: List[str]  # Movie titles
    rec_item: List[str]  # Movie titles

@dataclass
class UserData:
    """Represents processed user data with mapped movie titles."""
    user_id: str
    numerical_id: int
    history_interaction: List[str]  # Movie titles
    user_might_like: List[str]  # Movie titles
    conversations: List[ConversationTurn]
    raw_conversation_text: Optional[str] = None

class MovieRecommenderDataLoader:
    """
    Data loader class that processes and combines movie recommendation datasets
    with explicit path handling and improved conversation parsing.
    """
    
    def __init__(
        self,
        jsonl_path: str,
        user_ids_path: str,
        item_map_path: str,
        conversation_path: str
    ):
        """
        Initialize the data loader with explicit paths to all required files.
        
        Args:
            jsonl_path: Path to the main dataset JSONL file
            user_ids_path: Path to the user IDs mapping file
            item_map_path: Path to the item mapping file
            conversation_path: Path to the conversation text file
        """
        self.jsonl_path = Path(jsonl_path)
        self.user_ids_path = Path(user_ids_path)
        self.item_map_path = Path(item_map_path)
        self.conversation_path = Path(conversation_path)
        
        self.item_map: Dict[str, str] = {}
        self.user_id_map: Dict[str, int] = {}
        self.conversations_text: Dict[int, str] = {}
        self.processed_data: Dict[str, UserData] = {}

    def validate_paths(self) -> None:
        """Validate that all required files exist and are readable."""
        for path, name in [
            (self.jsonl_path, "JSONL dataset"),
            (self.user_ids_path, "User IDs mapping"),
            (self.item_map_path, "Item mapping"),
            (self.conversation_path, "Conversation text")
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{name} file not found: {path}")
            if not path.is_file():
                raise ValueError(f"{name} is not a file: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    _ = f.read(1)
            except Exception as e:
                raise IOError(f"Cannot read {name} at {path}: {str(e)}")

    def load_item_map(self) -> None:
        """Load and process the item mapping file."""
        try:
            with open(self.item_map_path, 'r', encoding='utf-8') as f:
                self.item_map = json.load(f)
            logger.info(f"Loaded {len(self.item_map)} items from item map")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in item map file: {e}")
            raise

    def load_user_id_map(self) -> None:
        """Load and process the user ID mapping file."""
        try:
            with open(self.user_ids_path, 'r', encoding='utf-8') as f:
                self.user_id_map = json.load(f)
            logger.info(f"Loaded {len(self.user_id_map)} user IDs")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in user ID map file: {e}")
            raise

    def load_conversations(self) -> None:
        """Load and process the conversations file with improved parsing."""
        try:
            with open(self.conversation_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # Split the content by conversation boundaries
            raw_conversations = content.split('\n\n')
            current_conv_id = None
            current_text = []

            for chunk in raw_conversations:
                chunk = chunk.strip()
                if not chunk:
                    continue

                # If chunk is just a number, it's a conversation ID
                if chunk.isdigit():
                    # Save previous conversation if exists
                    if current_conv_id is not None and current_text:
                        self.conversations_text[current_conv_id] = '\n'.join(current_text)
                        current_text = []
                    current_conv_id = int(chunk)
                else:
                    # If we have a current conversation ID, append the text
                    if current_conv_id is not None:
                        current_text.append(chunk)

            # Save the last conversation if exists
            if current_conv_id is not None and current_text:
                self.conversations_text[current_conv_id] = '\n'.join(current_text)

            logger.info(f"Loaded {len(self.conversations_text)} conversations")
        except Exception as e:
            logger.error(f"Error loading conversations: {str(e)}")
            raise

    def get_movie_title(self, movie_id: str) -> str:
        """Get movie title from ID, with fallback to ID if not found."""
        return self.item_map.get(movie_id, f"Unknown Title (ID: {movie_id})")

    def create_mapped_dataset(self) -> List[Dict[str, Any]]:
        """Create fully mapped dataset with all movie IDs replaced by titles."""
        mapped_data = []
        
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    user_data = json.loads(line)
                    for user_id, data in user_data.items():
                        # Create base user data structure
                        mapped_user_data = {
                            "user": {
                                "amazon_id": user_id,
                                "numerical_id": self.user_id_map.get(user_id, -1)
                            },
                            "movies": {
                                "history": [
                                    {
                                        "id": movie_id,
                                        "title": self.get_movie_title(movie_id)
                                    }
                                    for movie_id in data['history_interaction']
                                ],
                                "recommendations": [
                                    {
                                        "id": movie_id,
                                        "title": self.get_movie_title(movie_id)
                                    }
                                    for movie_id in data['user_might_like']
                                ]
                            },
                            "conversations": []
                        }

                        # Process each conversation
                        for conv in data['Conversation']:
                            conv_data = next(iter(conv.values()))
                            conv_id = conv_data['conversation_id']
                            
                            mapped_conv = {
                                "conversation_id": conv_id,
                                "movies": {
                                    "liked": [
                                        {
                                            "id": movie_id,
                                            "title": self.get_movie_title(movie_id)
                                        }
                                        for movie_id in conv_data['user_likes']
                                    ],
                                    "disliked": [
                                        {
                                            "id": movie_id,
                                            "title": self.get_movie_title(movie_id)
                                        }
                                        for movie_id in conv_data['user_dislikes']
                                    ],
                                    "recommended": [
                                        {
                                            "id": movie_id,
                                            "title": self.get_movie_title(movie_id)
                                        }
                                        for movie_id in conv_data['rec_item']
                                    ]
                                }
                            }
                            
                            # Add conversation text if available
                            if conv_id in self.conversations_text:
                                mapped_conv["text"] = self.conversations_text[conv_id]
                            else:
                                mapped_conv["text"] = "Conversation text not found"
                            
                            mapped_user_data["conversations"].append(mapped_conv)
                        
                        mapped_data.append(mapped_user_data)
                        
            return mapped_data
        
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

    def export_mapped_dataset(self, output_path: str) -> None:
        """Export the fully mapped dataset to a JSONL file."""
        try:
            mapped_data = self.create_mapped_dataset()
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for entry in mapped_data:
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')
                    
            logger.info(f"Successfully exported mapped dataset to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting mapped dataset: {str(e)}")
            raise

    def process_all(self, output_path: str) -> None:
        """Process all data and export the mapped dataset."""
        try:
            self.validate_paths()
            self.load_item_map()
            self.load_user_id_map()
            self.load_conversations()
            self.export_mapped_dataset(output_path)
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            raise

def main():
    """Main execution function with explicit paths and improved error handling."""
    try:
        # Define explicit paths
        paths = {
            'jsonl_path': './LLM_Redial/Movie/final_data.jsonl',
            'user_ids_path': './LLM_Redial/Movie/user_ids.json',
            'item_map_path': './LLM_Redial/Movie/item_map.json',
            'conversation_path': './LLM_Redial/Movie/Conversation.txt'
        }
        
        # Initialize data loader with explicit paths
        loader = MovieRecommenderDataLoader(**paths)
        
        # Set output path
        output_path = './LLM_Redial/Movie/fully_mapped_user_ids_and_movie_ids.jsonl'
        
        # Process and export the mapped dataset
        loader.process_all(output_path)
        
        logger.info("Data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()