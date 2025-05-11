import numpy as np


def householder_bidiag_snapshots(A: np.ndarray):
    """
    Bidiagonalises an m x n matrix A with Householder reflections

    Returns
    -------
    list[list[np.ndarray]]
        [
          [B₀, B₁, …, Bₖ],   # bidiagonal matrix after each step
          [V₀, V₁, …, Vₖ],   # accumulating right reflectors
          [U₀, U₁, …, Uₖ],   # accumulating left  reflectors
        ]
        where k = min(m, n)
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    B = A.copy()
    U = np.eye(m)
    V = np.eye(n)

    # history lists – include the *initial* state as snapshot 0
    B_hist = [B.copy()]
    V_hist = [V.copy()]
    U_hist = [U.copy()]

    for k in range(min(m, n)):
        # LEFT reflector (zero below B[k, k])
        v = B[k:, k].copy()
        norm_v = np.linalg.norm(v)
        if norm_v != 0:
            v[0] += np.sign(v[0]) * norm_v
            beta = 2.0 / (v @ v)

            # Apply to B[k:, k:]
            B[k:, k:] -= beta * np.outer(v, v @ B[k:, k:])

            # Accumulate in U  (U ← U · Hᵀ because we apply on the left)
            U[:, k:] -= beta * np.outer(U[:, k:] @ v, v)

        # RIGHT reflector (zero right of B[k, k])
        if k < n - 1:
            v2 = B[k, k + 1 :].copy()
            norm_v2 = np.linalg.norm(v2)
            if norm_v2 != 0:
                v2[0] += np.sign(v2[0]) * norm_v2
                beta2 = 2.0 / (v2 @ v2)

                # Apply to B[k:, k+1:]
                B[k:, k + 1 :] -= beta2 * np.outer(B[k:, k + 1 :] @ v2, v2)

                # Accumulate in V (V ← V · Hᵀ because we apply on the right)
                V[:, k + 1 :] -= beta2 * np.outer(V[:, k + 1 :] @ v2, v2)

        # stash snapshots AFTER finishing column k
        B_hist.append(B.copy())
        V_hist.append(V.copy())
        U_hist.append(U.copy())

    return [B_hist, V_hist, U_hist]
