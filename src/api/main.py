from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.router import route
from models.api_model import generate as generate_api

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

def generate_local_mock(prompt: str) -> str:
    return f"[réponse simulée du petit modèle local pour: '{prompt[:50]}...']"

@app.post("/chat")
def chat(request: QueryRequest):
    decision = route(request.query)
    
    if decision == "local":
        response = generate_local_mock(request.query)
    else:
        response = generate_api(request.query)
    
    return {
        "query": request.query,
        "model_used": decision,
        "response": response
    }