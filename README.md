# PROJET : MODEL ROUTER

## Où en est le projet - bilan complet

### Objectif

Construire un orchestrateur de routage de requêtes entre un petit modèle de langage tournant en local (rapide, gratuit) et un gros modèle appelé via API (plus cher, plus performant), avec mesure de coûts/latence à l'appui, pour démontrer qu'un routage intelligent permet de réduire les coûts tout en conservant une qualité de réponse acceptable.

### Choix structurants posés au départ

- **Interface de communication simple** entre les briques : une fonction qui prend un texte et retourne un texte, pour que chaque partie du système soit développée et testée indépendamment.
- **Structure de projet** : `src/models/` pour l'inférence (local et API), `src/router/` pour la logique de décision et de confiance, `src/evaluation/` pour les mesures de qualité et de performance, `src/logging/` pour la persistance, `src/api/` pour l'exposition réseau, `dashboard/` pour la visualisation.
- **Base SQLite** pour le logging plutôt qu'un système plus lourd - suffisant pour tracer les requêtes traitées et leurs métriques sans infrastructure supplémentaire.
- **FastAPI** pour exposer le système en service web, avec un développement en parallèle possible grâce à une fonction `generate_local` factice tant que le vrai modèle local n'est pas branché.
- **Config retenue pour le modèle local : Qwen2.5-3B-Instruct en fp16, sans quantization** - décision prise après comparaison de plusieurs tailles et configurations (voir ci-dessous), la quantization s'étant révélée contre-productive sur le GPU disponible (latence multipliée par 2 à 5, alors que la mémoire GPU disponible rend l'économie inutile).

### Ce qui est terminé et testé

**`src/models/local_model.py`** - `load_model()`/`generate_local(prompt)` fonctionnels. Chargement en cache (singleton) pour éviter de recharger le modèle à chaque appel, formatage via `apply_chat_template` pour respecter le format attendu par un modèle Instruct. Testé sur plusieurs prompts variés (question factuelle, calcul, demande créative), réponses cohérentes obtenues.

**`src/models/model_comparison.py`** - comparatif complet de 5 configurations (0.5B, 1.5B, 3B en fp16, 3B en 8-bit, 3B en 4-bit) sur latence, mémoire GPU et qualité qualitative des réponses. Résultat retenu : le 3B en fp16 offre le meilleur compromis sur la machine utilisée (latence la plus basse du lot pour le 3B, mémoire GPU disponible en suffisance, quantization pénalisant fortement la latence sans bénéfice réel ici).

**`src/router/router.py`** - première version du routage par règles simples : mots-clés associés à des requêtes simples (`bonjour`, `merci`, `traduis`...) ou complexes (`explique en détail`, `code`, `analyse`, `compare`...), complétés par un seuil de longueur (plus de 200 caractères → routage vers l'API). Testé sur un jeu de requêtes variées via son propre point d'entrée.

**`src/logging/logger.py`** - base SQLite fonctionnelle (`init_db`, `log_request`) avec une table `requests` (timestamp, requête, modèle utilisé, réponse, latence en ms, coût estimé). Testée avec une insertion de ligne factice.

**`src/api/main.py`** - service FastAPI avec un point d'entrée `POST /chat` : reçoit une requête, appelle `route()`, génère une réponse (actuellement via `generate_local_mock`, une fonction factice, côté local - l'appel API réel via `generate_api` est déjà branché côté gros modèle), mesure la latence, journalise dans la base, renvoie le résultat en JSON.

### Ce qui reste à faire

- **Brancher `generate_local` (le vrai modèle, déjà fonctionnel dans `local_model.py`) à la place de `generate_local_mock`** dans `main.py`.
- **Calcul réel du coût estimé** pour les appels API - actuellement une valeur fixe provisoire (0.0001), à remplacer par un calcul basé sur les tokens réellement consommés.
- **`src/router/classifier.py`** - classifieur de complexité appris (par embeddings ou fine-tuning), pour remplacer ou compléter les règles simples actuelles. Pas encore commencé.
- **`src/router/fallback.py`** - mécanisme de confiance et d'escalade automatique vers l'API en cas de réponse peu fiable du modèle local. Pas encore commencé.
- **`src/evaluation/llm_judge.py`** - évaluation de la qualité des réponses par un LLM-juge. Fichier vide pour l'instant.
- **`src/evaluation/benchmark.py`** - comparaison finale du système assemblé (tout local vs tout API vs routage intelligent) sur un jeu de requêtes annotées. Fichier vide pour l'instant.
- **`dashboard/app.py`** - interface Streamlit de visualisation des résultats (répartition par modèle, coût cumulé, latence moyenne). Fichier vide pour l'instant.
- **`data/test_queries.jsonl`** - jeu de requêtes annotées pour l'évaluation, à constituer (30-50 requêtes simple/complexe).