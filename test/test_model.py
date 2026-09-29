from src.models.local_model import generate_local

prompts = [
    "Quelle est la capitale de la France ?",
    "Calcule 15 fois 7.",
    "Raconte-moi une blague."
]

for p in prompts:
    print("Prompt :", p)
    print("Réponse :", generate_local(p))
    print("---")