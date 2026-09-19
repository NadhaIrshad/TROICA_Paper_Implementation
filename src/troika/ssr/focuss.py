"""Regularized M-FOCUSS (paper Section III-C, ref. [28] Cotter et al. 2005).

Paper Section IV-B: ``p = 0.8``, regularization parameter ``lambda = 0.1``,
grid ``N = 4096``, and 5 iterations, because coefficients with large values
converge after only a few iterations and full convergence is not needed.

The single-measurement-vector iteration is

    W_k = diag( |x_{k-1}|^(1 - p/2) )
    q_k = (Phi W_k)^H ( Phi W_k (Phi W_k)^H + lambda I_M )^-1 y
    x_k = W_k q_k

Solving that literally inverts an ``M x M`` matrix. After pruning there are 458
columns against ``M' = 998`` rows, so the push-through identity gives the same
answer from an ``n x n`` solve, which is roughly five times cheaper:

    A_k = (w w^T) * G + lambda I_n        with  w = |x_{k-1}|^(1 - p/2)
    q_k = solve(A_k, w * (Phi^H y))
    x_k = w * q_k

Both forms are implemented; :func:`focuss_reference` exists so the fast one can
be tested against it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

__all__ = ["focuss", "focuss_reference", "initial_x"]


def _solve_hermitian(A: NDArray[np.complex128], b: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Solve ``A q = b`` for Hermitian positive-definite ``A``.

    ``G = Phi^H Phi`` is Hermitian positive semi-definite, the Schur product
    with ``w w^T`` keeps it so, and ``+ lambda I`` makes it positive definite.
    Cholesky is therefore valid and about twice as fast as a general solve. The
    general solve remains as a fallback in case rounding pushes a near-zero
    weight vector outside the definite cone.
    """
    try:
        return cho_solve(cho_factor(A, lower=True, check_finite=False), b, check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.solve(A, b)


def initial_x(
    y: NDArray[np.float64],
    Phi: NDArray[np.complex128],
    mode: str = "ones",
) -> NDArray[np.complex128]:
    """Starting point of the iteration (ASSUMPTION A11).

    ``ones`` is the default and treats every kept bin as equally likely.
    ``matched_filter`` starts from ``|Phi^H y|``, which front-loads the
    periodogram's answer and converges faster but biases the result towards it.
    """
    n = Phi.shape[1]
    if mode == "ones":
        return np.ones(n, dtype=np.complex128)
    if mode == "matched_filter":
        return np.abs(Phi.conj().T @ y).astype(np.complex128)
    raise ValueError(f"unknown x0 mode {mode!r}; use 'ones' or 'matched_filter'")


def _weights(x: NDArray[np.complex128], p: float) -> NDArray[np.float64]:
    """``w = |x|^(1 - p/2)``, the diagonal of ``W_k``."""
    return np.abs(x) ** (1.0 - float(p) / 2.0)


def focuss(
    y: NDArray[np.float64],
    Phi: NDArray[np.complex128],
    G: NDArray[np.complex128],
    p: float = 0.8,
    lam: float = 0.1,
    n_iter: int = 5,
    x0: NDArray[np.complex128] | str = "ones",
    *,
    return_iterates: bool = False,
) -> NDArray[np.complex128] | tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Regularized M-FOCUSS in the fast ``n x n`` form.

    ``G`` must be ``Phi^H Phi``. With ``return_iterates`` the second return value
    has shape ``(n_iter + 1, n)`` and holds ``x0`` followed by every iterate,
    which is what ``plot_ssr_iterations`` displays.
    """
    y = np.asarray(y, dtype=np.float64)
    x = initial_x(y, Phi, x0) if isinstance(x0, str) else np.asarray(x0, dtype=np.complex128)

    b = Phi.conj().T @ y
    n = Phi.shape[1]
    eye = np.eye(n, dtype=np.complex128)
    iterates = [x.copy()]

    for _ in range(int(n_iter)):
        w = _weights(x, p)
        A = (np.outer(w, w) * G) + float(lam) * eye
        q = _solve_hermitian(A, (w * b).astype(np.complex128))
        x = w * q
        if return_iterates:
            iterates.append(x.copy())

    if return_iterates:
        return x, np.stack(iterates)
    return x


def focuss_reference(
    y: NDArray[np.float64],
    Phi: NDArray[np.complex128],
    p: float = 0.8,
    lam: float = 0.1,
    n_iter: int = 5,
    x0: NDArray[np.complex128] | str = "ones",
) -> NDArray[np.complex128]:
    """The literal ``M x M`` iteration, used only to test :func:`focuss`.

    This is the algorithm exactly as written in the paper, with no push-through
    identity applied. It is far slower and is never used in the pipeline.
    """
    y = np.asarray(y, dtype=np.float64)
    x = initial_x(y, Phi, x0) if isinstance(x0, str) else np.asarray(x0, dtype=np.complex128)

    M = Phi.shape[0]
    eye = np.eye(M, dtype=np.complex128)

    for _ in range(int(n_iter)):
        w = _weights(x, p)
        PW = Phi * w[None, :]
        inner = PW @ PW.conj().T + float(lam) * eye
        q = PW.conj().T @ np.linalg.solve(inner, y.astype(np.complex128))
        x = w * q
    return x
