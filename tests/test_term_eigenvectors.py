import numpy as np

from prf.strategies.prf_rerank_terms import term_space_eigenvectors, term_space_pca


def _vector_from_sparse(
    sparse_vector: dict[str, float], terms: list[str]
) -> np.ndarray:
    return np.array([sparse_vector.get(term, 0.0) for term in terms])


def test_term_space_eigenvectors_diagonal():
    term_to_importance = {
        "alpha": [(0, 1.0)],
        "beta": [(1, 2.0)],
    }
    terms = sorted(term_to_importance)

    eigenvalues, vectors = term_space_eigenvectors(term_to_importance, top_r=2)

    assert len(eigenvalues) == 2
    assert len(vectors) == 2
    vec0 = _vector_from_sparse(vectors[0], terms)
    vec1 = _vector_from_sparse(vectors[1], terms)

    assert np.isclose(np.dot(vec0, vec1), 0.0, atol=1e-8)
    assert np.isclose(np.linalg.norm(vec0), 1.0, atol=1e-8)
    assert np.isclose(np.linalg.norm(vec1), 1.0, atol=1e-8)
    assert abs(vec0[terms.index("beta")]) > abs(vec0[terms.index("alpha")])


def test_term_space_eigenvectors_correlated_terms():
    term_to_importance = {
        "alpha": [(0, 1.0), (1, 1.0)],
        "beta": [(0, 1.0), (1, 1.0)],
    }
    terms = sorted(term_to_importance)

    eigenvalues, vectors = term_space_eigenvectors(term_to_importance, top_r=2)

    assert len(eigenvalues) == 2
    assert len(vectors) == 2
    vec0 = _vector_from_sparse(vectors[0], terms)
    vec1 = _vector_from_sparse(vectors[1], terms)

    assert np.isclose(abs(vec0[0]), abs(vec0[1]), atol=1e-8)
    assert vec0[0] * vec0[1] > 0
    assert np.isclose(np.dot(vec0, vec1), 0.0, atol=1e-8)


def test_term_space_pca_centered_direction():
    term_to_importance = {
        "alpha": [(0, 1.0), (1, 2.0), (2, 3.0)],
        "beta": [(0, 1.0), (1, 2.0), (2, 3.0)],
    }
    terms = sorted(term_to_importance)

    eigenvalues, vectors = term_space_pca(term_to_importance, top_r=2)

    assert len(eigenvalues) == 2
    vec0 = _vector_from_sparse(vectors[0], terms)
    vec1 = _vector_from_sparse(vectors[1], terms)

    assert eigenvalues[0] >= eigenvalues[1]
    assert np.isclose(abs(vec0[0]), abs(vec0[1]), atol=1e-8)
    assert vec0[0] * vec0[1] > 0
    assert np.isclose(np.dot(vec0, vec1), 0.0, atol=1e-8)
