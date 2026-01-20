import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_groq import ChatGroq
from time import time
from prompts import get_enhanced_prompt

load_dotenv()

google_genai_api_key = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

class Searcher:
    def __init__(self, provider: str, api_key: str, model: str, use_tavily: bool = False):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self.use_tavily = use_tavily
        
        if self.provider == "google":
            if not api_key:
                raise ValueError("Google API key is required for Google provider")
            self.llm = GoogleGenerativeAI(api_key=api_key, model=model)
        elif self.provider == "groq":
            if not api_key:
                raise ValueError("Groq API key is required for Groq provider")
            self.llm = ChatGroq(api_key=api_key, model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: 'google', 'groq'")
        
        if use_tavily and tavily_api_key:
            self.tavily_search = TavilySearch(
                max_results=5,
                api_key=tavily_api_key
            )
            self.tavily_retriever = TavilySearchAPIRetriever(
                k=3,
                api_key=tavily_api_key
            )
        elif use_tavily and not tavily_api_key:
            raise ValueError("Tavily API key is required when use_tavily=True")

    def invoke(self, prompt: str):
        return self.llm.invoke(prompt)
    
    def search(self, query: str):
        if not self.use_tavily:
            raise ValueError("Tavily is not enabled. Set use_tavily=True in __init__")
        return self.tavily_search.invoke({"query": query})
    
    def invoke_with_search(self, question: str):  
        if not self.use_tavily:
            return self.invoke(question)
        
        search_start_time = time()
        search_output = self.tavily_search.invoke({"query": question})
        search_end_time = time()
        print(f"Time taken to search: {search_end_time - search_start_time} seconds")
      
        if isinstance(search_output, dict):
            results_list = search_output.get("results", [])
        elif isinstance(search_output, list):
            results_list = search_output
        else:
            results_list = []
            context = str(search_output)
        
        if results_list:
            context = "\n\n".join([
                f"Title: {result.get('title', 'N/A')}\nContent: {result.get('content', 'N/A')}"
                for result in results_list
                if isinstance(result, dict)
            ])
        else:
            context = str(search_output)
        
        inference_start_time = time()
        inference_output = self.llm.invoke(get_enhanced_prompt(context, question))
        inference_end_time = time()
        print(f"Time taken to infer: {inference_end_time - inference_start_time} seconds")
        if inference_output.content:
            return inference_output.content
        else:
            return inference_output


if tavily_api_key and groq_api_key:
    groq_llm = Searcher(
        provider="groq",
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",  
        use_tavily=True
    )
    
    print("Using Groq with Tavily Search:")
    search_answer = groq_llm.invoke_with_search("What are the top facts about the company 2care AI?")
    print(search_answer)
    print("\n")
else:
    print("Tavily or Groq API key is not set")
