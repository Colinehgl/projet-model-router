import numpy as np
import pytest

from src.router.classifier import (EmbeddingClassifier, load_dataset, normalize_label,)

SIMPLE = ["Bonjour !", "Quelle est la capitale de la France ?", "Merci beaucoup", "Il fait quel temps ?"]
COMPLEX = [
    "Démontre que la série harmonique diverge puis généralise aux séries de Riemann.",
    "Écris un décorateur Python thread-safe qui met en cache avec expiration et explique sa complexité.",
    "Compare les architectures Transformer et Mamba en détaillant les compromis mémoire/latence.",
    "Analyse les implications économiques d'une sortie de la zone euro sur la dette souveraine.",
]


def test_normalize_label():
    assert normalize_label("simple") == 0
    assert normalize_label("Local") == 0
    assert normalize_label("complexe") == 1
    assert normalize_label("api") == 1
    with pytest.raises(ValueError):
        normalize_label("bof")


def test_load_dataset(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        '{"query": "a", "label": "simple"}\n\n{"prompt": "b", "complexity": "complexe"}\n',
        encoding="utf-8",
    )
    texts, y = load_dataset(p)
    assert texts == ["a", "b"] and list(y) == [0, 1]


def test_fit_requires_both_classes():
    clf = EmbeddingClassifier()
    with pytest.raises(ValueError):
        clf.fit_embeddings(np.random.rand(3, 4), [0, 0, 0])


@pytest.mark.parametrize("backend", ["logreg", "knn"])
def test_roundtrip_and_route(tmp_path, backend):
    clf = EmbeddingClassifier(backend=backend).fit(SIMPLE + COMPLEX, [0] * 4 + [1] * 4)
    p = clf.predict_proba("Merci, bonne journée !")
    assert 0.0 <= p <= 1.0
    assert clf.route("Bonjour, ça va ?") in ("local", "api")

    path = tmp_path / "clf.joblib"
    clf.save(path)
    clf2 = EmbeddingClassifier.load(path)
    assert clf2.predict_proba("Merci, bonne journée !") == pytest.approx(p)