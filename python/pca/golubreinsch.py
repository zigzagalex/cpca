import math

import numpy as np
from householder import householder_bidiag_snapshots


def givens_rotation(a: float, b: float, tol: float = 1e-16):
    """Return cos (c) and -sin (s) that zero-out `b` when applied to (a, b)."""
    if abs(b) < tol:
        return 1.0, 0.0
    if abs(b) > abs(a):
        r = math.hypot(a, b)
        return a / r, -b / r
    t = -b / a
    c = 1.0 / math.sqrt(1.0 + t * t)
    return c, c * t


def apply_givens_to_cols(M: np.ndarray, i: int, j: int, c: float, s: float):
    """M ← M · Rᵀ  (acts on *columns* i & j)."""
    Mi, Mj = M[:, i].copy(), M[:, j].copy()
    M[:, i] = c * Mi - s * Mj
    M[:, j] = s * Mi + c * Mj


def apply_givens_to_rows(M: np.ndarray, i: int, j: int, c: float, s: float):
    """M ← R · M   (acts on *rows* i & j)."""
    Mi, Mj = M[i, :].copy(), M[j, :].copy()
    M[i, :] = c * Mi - s * Mj
    M[j, :] = s * Mi + c * Mj


def compute_shift(B: np.ndarray, q: int):
    """
    Wilkinson shift mufor the trailing 2x2 of BᵀB - returns √λ (see original C).
    `q` is the *last* diagonal index of the active block (q ≥ 1).
    """
    f, g, h = B[q - 1, q - 1], B[q - 1, q], B[q, q]
    a11, a12, a22 = f * f + g * g, f * g, h * h + g * g
    tr, det = a11 + a22, a11 * a22 - a12 * a12
    disc = math.sqrt(max(0.0, 0.25 * tr * tr - det))
    λ1, λ2 = 0.5 * tr + disc, 0.5 * tr - disc
    λ = λ1 if abs(λ1 - a22) < abs(λ2 - a22) else λ2
    return math.sqrt(max(0.0, λ))


def find_active_block(B: np.ndarray):
    """
    Return (p, q) so that B[p:p+1, p+1]…B[q-1,q] are the *only* non-zero
    super-diagonal elements in the current bidiagonal i.e. [p,q] is one
    continuous “active” Golub-Kahan block.
    If the matrix is already diagonal → (min_mn, 0)
    """
    min_mn = min(B.shape)
    p = 0
    while p < min_mn - 1 and B[p, p + 1] == 0:  # skip zero bands
        p += 1
    q = p
    while q < min_mn - 1 and B[q, q + 1] != 0:  # ride non-zero band
        q += 1
    return p, q


def golub_reinsch_svd_snapshots(A: np.ndarray, epsilon=1e-12, max_iter=10_000):
    """
    Full Golub-Reinsch SVD translation with *history*:
    returns [B_snapshots, V_snapshots, U_snapshots]
    """
    A = np.asarray(A, dtype=float)
    m0, n0 = A.shape
    wide = n0 > m0  # follow the same “do it on Aᵀ if wide” trick
    A_work = A.T.copy() if wide else A.copy()
    m, n = A_work.shape

    # 1) Bidiagonalise
    (
        B_hist,
        V_hist,
        U_hist,
    ) = householder_bidiag_snapshots(A_work)
    B = B_hist[-1]
    V = V_hist[-1]
    U = U_hist[-1]
    min_mn = min(m, n)

    # 2) QR steps on bidiagonal
    p, q = find_active_block(B)
    while p < min_mn and q != 0:  # 0 means “already diagonal”
        # trivial 1×1 block → skip
        if p == q:
            p = q + 1
            if p < min_mn:
                p, q = find_active_block(B)
            continue

        it = 0
        while True:
            # (2.1) annihilate tiny super-diagonals
            for i in range(p, q):
                if abs(B[i, i + 1]) <= epsilon * (abs(B[i, i]) + abs(B[i + 1, i + 1])):
                    B[i, i + 1] = 0.0

            # (2.2) has the block split?
            split_at = next((i for i in range(p, q) if B[i, i + 1] == 0.0), None)
            if split_at is not None:
                break  # go back to outer while – block boundaries changed

            if it >= max_iter:
                raise RuntimeError(f"SVD failed to converge for block [{p},{q}]")
            it += 1

            # (2.5) special case: zero diagonal within block
            handled = False
            for i in range(p, q):
                if B[i, i] == 0:
                    c, s = givens_rotation(0.0, B[i, i + 1])
                    apply_givens_to_rows(B, i, i + 1, c, s)
                    apply_givens_to_cols(U, i, i + 1, c, s)
                    handled = True
                    break
            if handled:
                B_hist.append(B.copy())
                V_hist.append(V.copy())
                U_hist.append(U.copy())
                continue

            # (2.6) Golub-Kahan “bulge chasing”
            mu = compute_shift(B, q)
            alpha, beta = B[p, p], B[p, p + 1]
            c, s = givens_rotation(alpha * alpha - mu * mu, alpha * beta)

            # push first rotation
            apply_givens_to_cols(B, p, p + 1, c, s)
            apply_givens_to_cols(V, p, p + 1, c, s)

            # chase the bulge down & right
            for k in range(p, q):
                # left rotation (rows k,k+1)
                c, s = givens_rotation(B[k, k], B[k + 1, k])
                apply_givens_to_rows(B, k, k + 1, c, s)
                apply_givens_to_cols(U, k, k + 1, c, s)

                # right rotation (cols k+1, k+2) – stay inside bidiagonal’s band
                if k < q - 1 and k + 2 < n:
                    c, s = givens_rotation(B[k, k + 1], B[k, k + 2])
                    apply_givens_to_cols(B, k + 1, k + 2, c, s)
                    apply_givens_to_cols(V, k + 1, k + 2, c, s)

            # remember after each full “bulge chase”
            B_hist.append(B.copy())
            V_hist.append(V.copy())
            U_hist.append(U.copy())

        # (re-)locate next active block
        p, q = find_active_block(B)

    # 3) fix wide-matrix case (swap roles of U & V)
    if wide:
        U, V = V, U  # *snapshots* must also swap
        U_hist = [V_snap for V_snap in V_hist]
        V_hist = [U_snap for U_snap in U_hist]

    return [B_hist, V_hist, U_hist]
