"""Routage des requêtes : "local" (petit modèle) ou "api" (gros modèle).

- route()          : règles simples (A2)
- route_learned()  : classifieur appris sur embeddings (B3), avec repli sur les règles
"""
import threading
from pathlib import Path

# ---------- Règles simples (A2) ----------
COMPLEX_KEYWORDS = (
    "explique en détail", "analyse", "code", "compare",
    "raisonne", "démontre", "pourquoi", "calcule",
)
SIMPLE_KEYWORDS = (
    "bonjour", "salut", "merci", "résume en une phrase", "traduis",
)
LENGTH_THRESHOLD = 200  # au-delà (en caractères) : "api"

# ---------- Routage appris (B3) ----------
# Racine du projet = deux dossiers au-dessus de src/router/router.py
TRAIN_PATH = Path(__file__).resolve().parents[2] / "data" / "routing_example.jsonl"

_learned = None
_failed = False
_lock = threading.Lock()


def route(query: str) -> str:
    """Routage par règles : les mots-clés « simples » l'emportent sur tout le reste."""
    query_lower = query.lower()

    if any(kw in query_lower for kw in SIMPLE_KEYWORDS):
        return "local"
    if len(query) > LENGTH_THRESHOLD or any(kw in query_lower for kw in COMPLEX_KEYWORDS):
        return "api"
    return "local"


def route_learned(query: str) -> str:
    """Routage appris (régression logistique sur embeddings).

    Entraîné à la première utilisation sur TRAIN_PATH, puis gardé en mémoire.
    Si quelque chose échoue (fichier absent, encodeur indisponible), le message s'affiche
    une seule fois et les règles sont utilisées jusqu'au redémarrage.
    """
    global _learned, _failed
    if _failed:
        return route(query)
    try:
        with _lock:
            if _learned is None:
                from src.router.classifier import EmbeddingClassifier, load_dataset

                texts, y = load_dataset(TRAIN_PATH)
                _learned = EmbeddingClassifier(backend="logreg").fit(texts, y)
        return _learned.route(query)
    except Exception as e:
        _failed = True
        print(f"[routage appris indisponible : {e}] règles utilisées")
        return route(query)


def main() -> None:
    test_queries = [
        "Bonjour, comment ça va ?",
        "Explique en détail les différences entre TCP et UDP",
        "Résume ce texte en une phrase : le chat dort sur le canapé.",
        "Écris-moi un algorithme de tri fusion en Python et analyse sa complexité",
        "Quelle heure est-il généralement pour le déjeuner en France ?",
        "Pourquoi le ciel est bleu ?",
        "Merci beaucoup pour ton aide !",
    ]

    for q in test_queries:
        rules, learned = route(q), route_learned(q)
        flag = "" if rules == learned else "  ← différent"
        print(f"[règles {rules.upper():5} | appris {learned.upper():5}] {q}{flag}")


if __name__ == "__main__":
    main()