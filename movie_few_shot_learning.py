from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import CTransformers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MovieExample:
    """Structure for few-shot examples"""
    user_input: str
    user_likes: List[str]
    user_dislikes: List[str]
    recommendation: str
    reasoning: str

class PureFewShotRecommender:
    """Movie recommender using pure few-shot learning without retrieval"""
    
    # Comprehensive examples covering different scenarios
    FEW_SHOT_EXAMPLES = [
        MovieExample(
            user_input="I love musicals with great performances. White Christmas was amazing!",
            user_likes=["White Christmas"],
            user_dislikes=[],
            recommendation="Holiday Inn",
            reasoning="Since you enjoyed White Christmas, Holiday Inn would be perfect. It's another classic musical featuring similar themes, great musical numbers, and strong performances. It actually inspired White Christmas and shares its festive atmosphere."
        ),
        MovieExample(
            user_input="The action in The Bourne Identity was incredible! Love movies that keep me on the edge of my seat.",
            user_likes=["The Bourne Identity"],
            user_dislikes=[],
            recommendation="Mission: Impossible",
            reasoning="Given your enthusiasm for The Bourne Identity's intense action, Mission: Impossible offers similar high-stakes thrills, complex plotting, and edge-of-your-seat action sequences."
        ),
        MovieExample(
            user_input="I really didn't like The Screaming Skull. Poor production quality and terrible acting.",
            user_likes=[],
            user_dislikes=["The Screaming Skull"],
            recommendation="Girl with a Pearl Earring",
            reasoning="Understanding your dislike of poor production quality, Girl with a Pearl Earring offers the opposite experience - beautifully crafted cinematography, exceptional acting from Scarlett Johansson and Colin Firth, and high production values."
        ),
        MovieExample(
            user_input="I enjoy thought-provoking dramas with strong character development.",
            user_likes=["The Lord of the Rings: The Fellowship of the Ring"],
            user_dislikes=["Driven"],
            recommendation="Master and Commander: The Far Side of the World",
            reasoning="Based on your appreciation for well-crafted narratives and character development, Master and Commander offers a compelling drama with rich characters, historical depth, and thoughtful storytelling."
        )
    ]

    def __init__(self, model_path: str, model_type: str = "llama"):
        """Initialize the recommender"""
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"  # Added memory_key
        )
        
        self.llm = CTransformers(
            model=model_path,
            model_type=model_type,
            config={
                "max_new_tokens": 512,
                "temperature": 0.7,
                "context_length": 2048,
                "gpu_layers": 0
            }
        )
        
        self.chain = self._create_chain()
    
    def _format_examples(self) -> str:
        """Format few-shot examples for the prompt"""
        formatted_examples = ""
        for i, example in enumerate(self.FEW_SHOT_EXAMPLES, 1):
            formatted_examples += f"""
Example {i}:
User: {example.user_input}
Liked Movies: {', '.join(example.user_likes) if example.user_likes else 'None'}
Disliked Movies: {', '.join(example.user_dislikes) if example.user_dislikes else 'None'}
Assistant: Based on your preferences, I recommend watching '{example.recommendation}'. {example.reasoning}

"""
        return formatted_examples.strip()
    
    def _create_chain(self):
        """Create the conversation chain"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are an expert movie recommendation assistant. 
Use these examples to understand how to provide personalized recommendations:

{self._format_examples()}

Guidelines:
1. Focus on understanding the user's stated preferences and movie experiences
2. Consider both what they enjoy and what they dislike
3. Provide a clear recommendation with thoughtful reasoning
4. Be conversational but informative
5. If unsure, ask clarifying questions
6. Draw parallels between recommended movies and user's stated preferences

Remember, your task is to recommend movies that match the user's taste while explaining your reasoning."""),
            
            HumanMessage(content="{input}")
        ])
        
        # Simple chain: prompt -> LLM -> output parser
        chain = prompt | self.llm | StrOutputParser()
        
        return chain
    
    async def generate_recommendation(self, user_input: str) -> Dict[str, Any]:
        """Generate a movie recommendation based on user input"""
        try:
            # Log the interaction
            logger.info(f"Processing user input: {user_input}")
            
            # Get conversation history
            history = self.memory.load_memory_variables({})
            
            # Generate response
            response = await self.chain.ainvoke({
                "input": user_input
            })
            
            # Update memory
            self.memory.save_context(
                {"input": user_input},
                {"output": response}
            )
            
            result = {
                "recommendation": response,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Generated recommendation: {response}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            raise
    
    def clear_history(self):
        """Clear conversation history"""
        self.memory.clear()
        logger.info("Conversation history cleared")

# FastAPI implementation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class RecommendationRequest(BaseModel):
    user_input: str

@dataclass
class APIConfig:
    model_path: str = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    model_type: str = "llama"

# Initialize recommender
config = APIConfig()
recommender = PureFewShotRecommender(config.model_path, config.model_type)

@app.post("/recommend/")
async def get_recommendation(request: RecommendationRequest):
    """Get movie recommendation"""
    try:
        return await recommender.generate_recommendation(request.user_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/")
async def clear_history():
    """Clear conversation history"""
    recommender.clear_history()
    return {"message": "Conversation history cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)