import os
import sys
import time

from fastapi import FastAPI

dossier_api = os.path.dirname(os.path.abspath(__file__))
dossier_src = os.path.dirname(dossier_api)
sys.path.append(dossier_src)

from router.router import route
from models.api_model import generate as generate_api
from request_logger.logger import init_db, log_request

app = FastAPI()
init_db()


def generate_local_mock(prompt):
    debut_du_prompt = prompt[:50]
    return "[réponse simulée du petit modèle local pour: '" + debut_du_prompt + "...']"


def chat(request: dict):
    query = request["query"]

    heure_debut = time.time()

    decision = route(query)

    if decision == "local":
        response = generate_local_mock(query)
        estimated_cost = 0.0
    else:
        response = generate_api(query)
        estimated_cost = 0.0001

    duree_secondes = time.time() - heure_debut
    latency_ms = duree_secondes * 1000

    log_request(
        query=query,
        model_used=decision,
        response=response,
        latency_ms=latency_ms,
        estimated_cost=estimated_cost,
    )

    resultat = {
        "query": query,
        "model_used": decision,
        "response": response,
        "latency_ms": round(latency_ms, 2),
    }
    return resultat


app.post("/chat")(chat)