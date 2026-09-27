from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.router import route
from models.api_model import generate as generate_api
from request_logger.logger import init_db, log_request

app = FastAPI()

init_db()

class QueryRequest(BaseModel):
    query: str

def generate_local_mock(prompt: str) -> str:
    return f"[réponse simulée du petit modèle local pour: '{prompt[:50]}...']"

@app.post("/chat")
def chat(request: QueryRequest):
    start_time = time.time()
    
    decision = route(request.query)
    
    if decision == "local":
        response = generate_local_mock(request.query)
        estimated_cost = 0.0
    else:
        response = generate_api(request.query)
        estimated_cost = 0.0001  # valeur provisoire, on affinera avec le vrai calcul de tokens
    
    latency_ms = (time.time() - start_time) * 1000
    
    log_request(
        query=request.query,
        model_used=decision,
        response=response,
        latency_ms=latency_ms,
        estimated_cost=estimated_cost
    )
    
    return {
        "query": request.query,
        "model_used": decision,
        "response": response,
        "latency_ms": round(latency_ms, 2)
    }