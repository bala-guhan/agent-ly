from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

class LLMProvider:
    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        if self.provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            # Configure ChatGroq for better tool calling reliability
            # Lower temperature for more deterministic tool calls
            # Groq's ChatGroq from langchain_groq is already optimized
            return ChatGroq(
                api_key=api_key,
                model=self.model,
                temperature=0.1,  # Lower temperature for more reliable tool calling
                max_retries=2  # Retry on failures
            )
        
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
    
    def invoke(self, prompt: str):
        response = self.llm.invoke(prompt)
        if hasattr(response, 'content'):
            return response.content
        return response
    
    def stream(self, prompt: str):
        for chunk in self.llm.stream(prompt):
            if hasattr(chunk, 'content'):
                yield chunk.content
            else:
                yield chunk
    
    def get_provider_info(self):
        return {
            "provider": self.provider,
            "model": self.model
        }

    