"""Tests for pilot kernels."""

import numpy as np

from brink.kernels import rbf


def test_rbf_symmetry_and_diagonal():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(30, 2))
    K = rbf(X, X, lengthscale=0.3)
    assert np.allclose(K, K.T, atol=1e-12)
    assert np.allclose(np.diag(K), 1.0, atol=1e-12)


def test_rbf_positive_semidefinite():
    rng = np.random.default_rng(2)
    X = rng.uniform(size=(60, 2))
    K = rbf(X, X, lengthscale=0.3)
    assert np.linalg.eigvalsh(K).min() >= -1e-10


def test_rbf_hand_checked_value():
    X1 = np.array([[0.0, 0.0]])
    X2 = np.array([[0.3, 0.4]])
    # ||x - x'||^2 = 0.25; k = exp(-0.25 / (2 * 0.5^2)) = exp(-0.5).
    K = rbf(X1, X2, lengthscale=0.5)
    assert np.allclose(K[0, 0], np.exp(-0.5), atol=1e-12)
