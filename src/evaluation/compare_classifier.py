from __future__ import annotations
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from src.router.classifier import (
    DEFAULT_ENCODER,
    EmbeddingClassifier,
    load_dataset,
    normalize_label,
)
import threading
from pathlib import Path

"""B3 — Comparaison règles simples (A2) vs classifieur appris sur embeddings.

Lancer depuis la racine du projet :
    python -m src.evaluation.compare_classifiers --data data/test_queries.jsonl \
        --save models/complexity_clf.joblib

Méthodologie
------------
- Les règles (A2) n'ont pas de paramètre appris : on les évalue directement sur tout le jeu.
- Le classifieur appris est évalué par validation croisée stratifiée répétée : chaque
  requête est prédite par un modèle qui ne l'a JAMAIS vue à l'entraînement (prédictions
  « out-of-fold »). Sans cela, la précision serait artificiellement gonflée.
- Les embeddings sont calculés une seule fois (l'encodeur est fixe, sans apprentissage).
- Métriques : accuracy, et surtout rappel de la classe « complexe » : une requête complexe
  envoyée au petit modèle dégrade la qualité, alors qu'une requête simple envoyée à l'API
  ne coûte que de l'argent.
- Avec 30 à 50 exemples, l'écart-type est grand : l'écart-type sur les répétitions ne mesure
  que la variabilité du découpage, pas l'incertitude due à la taille de l'échantillon.
"""


def load_rules_fn(spec: str | None = None):
    """Récupère la fonction de décision des règles simples.

    spec : "module:fonction", p. ex. "src.router.router:route_query".
    Sans spec, cherche un nom courant dans src.router.router puis src.router.classifier.
    La fonction doit prendre un texte et retourner 'local'/'api' (ou 'simple'/'complexe').
    """
    import importlib

    if spec:
        mod_name, _, fn_name = spec.partition(":")
        return getattr(importlib.import_module(mod_name), fn_name)

    names = ("classify", "classify_query", "classify_complexity", "decide", "route", "route_query")
    for mod_name in ("src.router.router", "src.router.classifier"):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for name in names:
            if hasattr(mod, name):
                return getattr(mod, name)
    raise ImportError("Fonction de règles introuvable : utilise --rules module:fonction")


def metrics(y_true, y_pred) -> dict:
    return {
        "acc": accuracy_score(y_true, y_pred),
        "recall_complex": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def cross_val_oof(texts, y, X, backend, n_splits, n_repeats, seed, **kw):
    """Retourne une liste (une par répétition) de prédictions out-of-fold."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof = [np.full(len(y), -1, dtype=int) for _ in range(n_repeats)]
    oof_proba = [np.zeros(len(y)) for _ in range(n_repeats)]
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        rep = i // n_splits
        clf = EmbeddingClassifier(backend=backend, **kw).fit_embeddings(X[tr], y[tr])
        p = clf.predict_proba_embeddings(X[te])
        oof_proba[rep][te] = p
        oof[rep][te] = (p >= clf.threshold).astype(int)
    return oof, oof_proba


def summarize(name, y, preds_list):
    ms = [metrics(y, p) for p in preds_list]
    out = {k: (np.mean([m[k] for m in ms]), np.std([m[k] for m in ms])) for k in ms[0]}
    print(
        f"{name:<22} acc = {out['acc'][0]:.3f} ± {out['acc'][1]:.3f} | "
        f"rappel complexe = {out['recall_complex'][0]:.3f} ± {out['recall_complex'][1]:.3f} | "
        f"F1 macro = {out['f1_macro'][0]:.3f} ± {out['f1_macro'][1]:.3f}"
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test_queries.jsonl")
    ap.add_argument("--save", default=None,
                    help="Optionnel : sauvegarde le meilleur modèle (.pt) entraîné sur tout le jeu")
    ap.add_argument("--rules", default=None,
                    help="Fonction des règles A2, format module:fonction (ex. src.router.router:route_query)")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER,
                    help="Modèle sentence-transformers (ex. all-MiniLM-L6-v2 pour comparer)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    texts, y = load_dataset(args.data)
    n_pos = int(y.sum())
    print(f"{len(y)} requêtes : {len(y) - n_pos} simples, {n_pos} complexes")
    n_splits = min(args.folds, int(min(n_pos, len(y) - n_pos)))
    if n_splits < 2:
        raise SystemExit("Pas assez d'exemples dans la classe minoritaire (< 2).")
    if n_splits < args.folds:
        print(f"[!] folds réduits à {n_splits} (taille de la classe minoritaire)")

    # --- Règles (A2)
    rules_fn = load_rules_fn(args.rules)
    y_rules = np.array([normalize_label(rules_fn(t)) for t in texts])
    print("\n=== Résultats ===")
    r = metrics(y, y_rules)
    print(
        f"{'Règles (A2)':<22} acc = {r['acc']:.3f}               | "
        f"rappel complexe = {r['recall_complex']:.3f}               | F1 macro = {r['f1_macro']:.3f}"
    )
    print("Matrice de confusion règles (lignes = vrai [simple, complexe]) :")
    print(confusion_matrix(y, y_rules, labels=[0, 1]))

    # --- Embeddings calculés une seule fois
    X = EmbeddingClassifier(encoder_name=args.encoder).embed(texts)

    configs = {
        "logreg C=1": ("logreg", {"C": 1.0}),
        "logreg C=10": ("logreg", {"C": 10.0}),
        "knn k=3": ("knn", {"k": 3}),
        "knn k=5": ("knn", {"k": 5}),
        "centroïdes": ("centroid", {}),
    }
    best_name, best_acc = None, -1.0
    for name, (backend, kw) in configs.items():
        oof, _ = cross_val_oof(texts, y, X, backend, n_splits, args.repeats, args.seed, **kw)
        out = summarize(name, y, oof)
        if out["acc"][0] > best_acc:
            best_name, best_acc = name, out["acc"][0]

    print(f"\nMeilleure configuration (accuracy CV) : {best_name}")
    print("[!] Choisir la config sur les mêmes données que celles qui servent à l'évaluer "
          "introduit un léger optimisme ; à garder en tête dans le rapport.")

    # --- Modèle final entraîné sur tout le jeu
    backend, kw = configs[best_name]
    if args.save:
        final = EmbeddingClassifier(backend=backend, encoder_name=args.encoder, **kw).fit_embeddings(X, y)
        final.save(args.save)
        print(f"Modèle sauvegardé : {args.save}")

    # --- Erreurs des règles vs du classifieur (analyse qualitative)
    oof, _ = cross_val_oof(texts, y, X, backend, n_splits, 1, args.seed, **kw)
    wrong_rules = {i for i in range(len(y)) if y_rules[i] != y[i]}
    wrong_learn = {i for i in range(len(y)) if oof[0][i] != y[i]}
    print(f"\nErreurs règles seulement : {len(wrong_rules - wrong_learn)} | "
          f"classifieur seulement : {len(wrong_learn - wrong_rules)} | "
          f"communes : {len(wrong_rules & wrong_learn)}")
    for i in sorted(wrong_learn):
        vrai = "complexe" if y[i] else "simple"
        print(f"  ✗ [{vrai}] {texts[i][:90]}")


if __name__ == "__main__":
    main()