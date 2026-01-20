import warnings
import sys
import uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agent.agent import EnterpriseAgent

warnings.filterwarnings("ignore")

app = FastAPI(title="Enterprise Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = None

@app.on_event("startup")
async def startup():
    global agent
    print("Initializing Enterprise Agent...")
    agent = EnterpriseAgent()
    print("✓ Agent ready!")


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    thread_id: str


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    thread_id = request.thread_id or str(uuid.uuid4())
    
    try:
        response = agent.chat(request.message, thread_id=thread_id)
        return ChatResponse(response=response, thread_id=thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history/{thread_id}")
async def get_history(thread_id: str):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        messages = agent.get_history(thread_id=thread_id)
        return {
            "thread_id": thread_id,
            "messages": [
                {
                    "type": msg.__class__.__name__,
                    "content": msg.content if hasattr(msg, "content") else str(msg)
                }
                for msg in messages
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

