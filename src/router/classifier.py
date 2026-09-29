from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
import torch
import torch.nn.functional as F

"""B3 — Classifieur de complexité appris

Principe : chaque requête est encodée par un modèle d'embeddings de phrases (chargé avec
`transformers`), puis une tête entraînée avec PyTorch prédit P(complexe | requête) :
    P(complexe) ≥ seuil  →  "api"   (gros modèle)
    P(complexe) <  seuil  →  "local" (petit modèle)

Trois têtes au choix (`backend`) :
    "logreg"   régression logistique (perte BCE pondérée par classe + régularisation L2)
    "knn"      k plus proches voisins en cosinus, votes pondérés par 1/distance
    "centroid" plus proche centroïde en cosinus

Usage :
    clf = EmbeddingClassifier.load("models/complexity_clf.pt")
    clf.route("Bonjour, ça va ?")            # -> "local"
    clf.predict_proba("Démontre que ...")     # -> 0.93
"""

DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_SIMPLE = {"simple", "local", "easy", "facile", "0"}
_COMPLEX = {"complex", "complexe", "api", "hard", "difficile", "1"}
_TEXT_KEYS = ("query", "prompt", "text", "question")
_LABEL_KEYS = ("label", "complexity", "complexite", "type", "category")

_BACKENDS = ("logreg", "knn", "centroid")


def normalize_label(raw) -> int:
    """'simple'/'local' -> 0 ; 'complexe'/'api' -> 1."""
    s = str(raw).strip().lower()
    if s in _SIMPLE:
        return 0
    if s in _COMPLEX:
        return 1
    raise ValueError(f"Étiquette inconnue : {raw!r}")


def load_dataset(path: str | Path):
    """Lit un .jsonl et retourne (textes, labels ∈ {0,1}). 
    de type tuple[list[str], np.ndarray]

    Clés acceptées pour le texte : query/prompt/text/question.
    Clés acceptées pour l'étiquette : label/complexity/complexite/type/category.
    """
    texts: list[str] = []
    labels: list[int] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tk = next((k for k in _TEXT_KEYS if k in row), None)
            lk = next((k for k in _LABEL_KEYS if k in row), None)
            if tk is None or lk is None:
                raise KeyError(
                    f"Ligne {i} : clés texte/étiquette introuvables (clés présentes : {list(row)})"
                )
            texts.append(row[tk])
            labels.append(normalize_label(row[lk]))
    return texts, np.asarray(labels, dtype=int)


def _unit(X):
    """Tableau (n, d) -> tenseur float64 dont les lignes sont de norme 1."""
    return F.normalize(torch.as_tensor(np.asarray(X), dtype=torch.float64), dim=1)


class EmbeddingClassifier:
    def __init__(
        self,
        backend: str = "logreg",
        encoder_name: str = DEFAULT_ENCODER,
        C: float = 1.0,
        k: int = 3,
        threshold: float = 0.5,
        max_length: int = 128,
        tau: float = 10.0,
    ):
        if backend not in _BACKENDS:
            raise ValueError(f"backend doit valoir l'une de {_BACKENDS}")
        self.backend = backend
        self.encoder_name = encoder_name
        self.C = C  # logreg : inverse de la force de régularisation (comme scikit-learn)
        self.k = k  # knn : nombre de voisins
        self.threshold = threshold
        self.max_length = max_length
        self.tau = tau  # centroid : température de la sigmoïde (probabilité NON calibrée)
        self._encoder = None
        self._device = None
        self._state: dict | None = None

    def _load_encoder(self):
        if self._encoder is None:
            from transformers import AutoModel, AutoTokenizer

            name = self.encoder_name
            if "/" not in name:  # ex. "all-MiniLM-L6-v2"
                name = f"sentence-transformers/{name}"
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name).to(self._device).eval()
            self._encoder = (tok, model)
        return self._encoder

    @torch.no_grad()
    def embed(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        """Embeddings de phrases : pooling moyen (masque d'attention) puis norme L2.

        Retourne un tableau float32 (n, d) ; cosinus = produit scalaire.
        """
        tok, model = self._load_encoder()
        texts = list(texts)
        if not texts:
            return np.zeros((0, model.config.hidden_size), dtype=np.float32)
        out = []
        for i in range(0, len(texts), batch_size):
            batch = tok(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            h = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(F.normalize(pooled, dim=-1).cpu())
        return torch.cat(out).numpy()

    # ---------- apprentissage des têtes ----------
    def _fit_logreg(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """Minimise (1/n)·Σ sᵢ·BCE(zᵢ, yᵢ) + ‖w‖² / (2·C·n), avec z = X·w + b.

        sᵢ = n / (2·n_classe(i)) : pondération « balanced ». Le biais n'est pas régularisé.
        Problème convexe, initialisé à 0, résolu par L-BFGS : résultat déterministe.
        """
        n = len(y)
        yf = y.to(torch.float64)
        n_pos = yf.sum()
        s = torch.where(y == 1, n / (2 * n_pos), n / (2 * (n - n_pos)))
        w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros((), dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS(
            [w, b],
            lr=1.0,
            max_iter=500,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            z = X @ w + b
            bce = F.binary_cross_entropy_with_logits(z, yf, reduction="none")
            loss = (s * bce).mean() + (w @ w) / (2 * self.C * n)
            loss.backward()
            return loss

        opt.step(closure)
        return {"w": w.detach(), "b": b.detach()}

    @staticmethod
    def _fit_knn(X: torch.Tensor, y: torch.Tensor) -> dict:
        return {"X": X, "y": y}

    @staticmethod
    def _fit_centroid(X: torch.Tensor, y: torch.Tensor) -> dict:
        C = torch.stack([X[y == c].mean(dim=0) for c in (0, 1)])
        return {"C": F.normalize(C, dim=1)}

    def fit_embeddings(self, X, y: Sequence[int]) -> "EmbeddingClassifier":
        y_t = torch.as_tensor(np.asarray(y, dtype=int))
        if set(y_t.unique().tolist()) != {0, 1}:
            raise ValueError("Le jeu d'entraînement doit contenir les deux classes.")
        fit = {"logreg": self._fit_logreg, "knn": self._fit_knn, "centroid": self._fit_centroid}
        self._state = fit[self.backend](_unit(X), y_t)
        return self

    def fit(self, texts: Sequence[str], y: Sequence[int]) -> "EmbeddingClassifier":
        return self.fit_embeddings(self.embed(texts), y)

    # ---------- inférence ----------
    def predict_proba_embeddings(self, X) -> np.ndarray:
        """P(complexe) pour chaque ligne de X (tableau (n, d))."""
        if self._state is None:
            raise RuntimeError("Classifieur non entraîné (appeler fit ou load).")
        X = _unit(X)
        st = self._state
        if self.backend == "logreg":
            p = torch.sigmoid(X @ st["w"] + st["b"])
        elif self.backend == "knn":
            k = min(self.k, len(st["y"]))
            dist = (1.0 - X @ st["X"].T).clamp(min=0.0)  # distance cosinus
            dk, idx = dist.topk(k, dim=1, largest=False)
            wts = 1.0 / dk.clamp(min=1e-12)  # vote pondéré par 1/distance
            p = (wts * st["y"][idx].to(torch.float64)).sum(dim=1) / wts.sum(dim=1)
        else:  # centroid : σ(τ·(s₁ − s₀)), s_c = cos(x, centroïde_c)
            s = X @ st["C"].T
            p = torch.sigmoid(self.tau * (s[:, 1] - s[:, 0]))
        return p.numpy()

    def predict_proba(self, text: str) -> float:
        return float(self.predict_proba_embeddings(self.embed([text]))[0])

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        p = self.predict_proba_embeddings(self.embed(texts))
        return (p >= self.threshold).astype(int)

    def route(self, text: str) -> str:
        """Interface compatible avec le routeur : 'local' ou 'api'."""
        return "api" if self.predict_proba(text) >= self.threshold else "local"

    # ---------- persistance ----------
    def save(self, path: str | Path) -> None:
        if self._state is None:
            raise RuntimeError("Rien à sauvegarder : classifieur non entraîné.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backend": self.backend,
                "encoder_name": self.encoder_name,
                "C": self.C,
                "k": self.k,
                "threshold": self.threshold,
                "max_length": self.max_length,
                "tau": self.tau,
                "state": self._state,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingClassifier":
        d = torch.load(path, map_location="cpu", weights_only=True)
        state = d.pop("state")
        obj = cls(**d)
        obj._state = state
        return obj