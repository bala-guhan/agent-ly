import warnings
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")
load_dotenv()


class Intent(BaseModel):
    intent: Literal["general", "2care_related"] = Field(
        description="Intent classification: 'general' for general queries, '2care_related' for queries about 2care company, products, or services"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )


class IntentClassifier:
    def __init__(
        self,
        llm_provider: str = "groq",
        llm_model: str = "llama-3.3-70b-versatile"
    ):
        self.provider = llm_provider.lower()
        self.model = llm_model
        self.llm = self._initialize_llm()
        # Use with_structured_output for automatic Pydantic parsing
        self.structured_llm = self.llm.with_structured_output(Intent)
    
    def _initialize_llm(self):
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            return ChatGroq(api_key=api_key, model=self.model)
        
        elif self.provider in ["google", "gemini"]:
            api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_GENERATIVE_AI_API_KEY not found in environment variables")
            return GoogleGenerativeAI(api_key=api_key, model=self.model)
        
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported providers: 'groq', 'google'"
            )
    
    def classify(self, query: str) -> Intent:
        """
        Classify user query intent using structured output.
        
        Args:
            query: User's query/question
            
        Returns:
            Intent object with intent, confidence, and reasoning
        """
        prompt = f"""You are an intent classification system. Analyze the user's query and determine if it relates to "2care" company, products, services, documentation, or business information.

Intent Types:
- "general": General questions not related to 2care (e.g., "What is the weather?", "Tell me a joke", "How do I cook pasta?")
- "2care_related": Questions about 2care company, products, services, documentation, technical details, company information, implementation, security, sales, customer success, etc.

Examples of 2care_related queries:
- "What is 2care?"
- "How does 2care API authentication work?"
- "Tell me about 2care's engineering team"
- "What are 2care's security compliance requirements?"
- "How do I implement 2care's product?"
- Any question mentioning "2care" or asking about company-specific information

User Query: "{query}"

Analyze this query and classify the intent. Provide your confidence score and reasoning.
"""
        
        # This automatically parses to Intent Pydantic model
        intent = self.structured_llm.invoke(prompt)
        return intent


def intent_classifier(query: str) -> Intent:
    classifier = IntentClassifier()
    return classifier.classify(query)
