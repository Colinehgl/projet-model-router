def route(query: str) -> str:
    complex_keywords = [
        "explique en détail", "analyse", "code", "compare", 
        "raisonne", "démontre", "pourquoi", "calcule"
    ]
    simple_keywords = [
        "bonjour", "salut", "merci", "résume en une phrase", "traduis"
    ]
    
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in simple_keywords):
        return "local"
    if len(query) > 200 or any(kw in query_lower for kw in complex_keywords):
        return "api"
    return "local"

if __name__ == "__main__":
    test_queries = [
        "Bonjour, comment ça va ?",
        "Explique en détail les différences entre TCP et UDP",
        "Résume ce texte en une phrase : le chat dort sur le canapé.",
        "Écris-moi un algorithme de tri fusion en Python et analyse sa complexité",
        "Quelle heure est-il généralement pour le déjeuner en France ?",
        "Pourquoi le ciel est bleu ?",
        "Merci beaucoup pour ton aide !"
    ]
    
    for q in test_queries:
        decision = route(q)
        print(f"[{decision.upper():5}] {q}")