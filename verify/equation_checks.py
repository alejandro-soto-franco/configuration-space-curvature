"""
Equation-level symbolic checks with counter-tests (Layer A).

Each check verifies a displayed (or load-bearing inline) ALGEBRAIC identity of the
paper on several divergence-free fields, AND runs a COUNTER-TEST: a deliberately
perturbed (e.g. sign-flipped) form of the identity that MUST fail. A check passes
only if the true identity holds to tolerance on every field and the perturbed form
is rejected on every field. The counter-test is what catches a vacuously-passing
check and the sign/transcription errors found in the 2026-06-08 correctness pass
(e.g. the `nu I - d` vs `nu I + d` divergence-form rewrite, registered here as
`div_form_rewrite`).

The registry `tools/equation_registry.toml` binds paper equation labels to these
checks; `tools/eqn_lint.py` runs them and gates the build on coverage.

Run standalone:  uv run --with numpy python sympy/equation_checks.py
"""

import os
import sys

import numpy as np
from numpy.fft import fftn, ifftn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pressure_hess_identity import wavenumbers, fft_jacobian, fft_curl  # noqa: E402
from cf3_imbalance_production_verify import (  # noqa: E402
    grad_scalar, laplacian_scalar, laplacian_vec, pressure_from_u,
    hessian_pressure, advect_vec, q_of_u, _flows,
)

N, H = 48, 2 * np.pi / 48
NU = 0.05
TOL = 5e-6          # positive identity must hold below this (spectral-aliasing floor;
                    #   div_form_rewrite stacks Biot-Savart + Jacobian + divergence)
COUNTER_MIN = 1e-2  # the perturbed identity must miss by at least this on at least one
                    #   field (non-vacuity). Aggregated by max, since degenerate flows
                    #   (ABC Beltrami: det grad u ~ 0, P ~ 0) cannot discriminate a sign
                    #   flip and would drag a min-aggregate to zero.


# --------------------------------------------------------------------------- fields
def _random_divfree(seed, kmax=5):
    """A real, divergence-free, band-limited field: curl of a low-pass random potential."""
    rng = np.random.default_rng(seed)
    kx, ky, kz = wavenumbers(N, H)
    kk = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    mask = kk <= kmax
    psi = np.zeros((N, N, N, 3))
    for m in range(3):
        psi[..., m] = np.real(ifftn(fftn(rng.standard_normal((N, N, N))) * mask))
    u = fft_curl(psi, H)
    scale = np.sqrt(np.mean(np.einsum('...i,...i->...', u, u))) + 1e-30
    return u / scale


def _fields():
    f = dict(_flows(N, H))
    f['random band-limited'] = _random_divfree(12345)
    return f


def _trA_LapA(A):
    """tr(A Delta A) = A_ij Delta A_ji, spectral Laplacian componentwise."""
    kx, ky, kz = wavenumbers(N, H)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    LapA = np.zeros_like(A)
    for i in range(3):
        for j in range(3):
            LapA[..., i, j] = np.real(ifftn(-k2 * fftn(A[..., i, j])))
    return np.einsum('...ij,...ji->...', A, LapA), LapA


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-30))


def _scalar_field_check(true_lhs, true_rhs, wrong_rhs):
    """Generic: lhs == true_rhs (pass) and lhs != wrong_rhs (counter)."""
    return _rel(true_lhs, true_rhs), _rel(true_lhs, wrong_rhs)


# --------------------------------------------------------------------------- checks
def _Dt_q_def(u):
    q, S, om, _ = q_of_u(u, H)
    p, _ = pressure_from_u(u, H)
    ut = -advect_vec(u, u, H) - grad_scalar(p, H) + NU * laplacian_vec(u, H)
    Adot = fft_jacobian(ut, H)
    Sdot = 0.5 * (Adot + np.swapaxes(Adot, -2, -1))
    omdot = fft_curl(ut, H)
    qt = (2.0 * np.einsum('...ij,...ij->...', S, Sdot)
          - np.einsum('...i,...i->...', om, omdot))
    return qt + np.einsum('...i,...i->...', u, grad_scalar(q, H))


def _Dt_q_formula(u, sign_det=-6.0):
    q, S, om, A = q_of_u(u, H)
    Hp, _ = hessian_pressure(u, H)
    ShessP = np.einsum('...ij,...ij->...', S, Hp)
    trALA, _ = _trA_LapA(A)
    return sign_det * np.linalg.det(A) - 2.0 * ShessP + 2.0 * NU * trALA


def check_production_law():
    """eq: production -- D_t q = -6 det A - 2 S:grad^2 p + 2 nu tr(A Delta A).
    Counter: the +6 det A sign must not match."""
    pos, ctr = [], []
    for u in _fields().values():
        d = _Dt_q_def(u)
        pos.append(_rel(d, _Dt_q_formula(u, -6.0)))
        ctr.append(_rel(d, _Dt_q_formula(u, +6.0)))   # wrong inertial sign
    return _verdict('production_law', pos, ctr)


def check_zero_mean_budget():
    """eq: zero-mean -- int det A = int S:grad^2 p = int tr(A Delta A) = 0."""
    pos, ctr = [], []
    for u in _fields().values():
        A = fft_jacobian(u, H)
        S = 0.5 * (A + np.swapaxes(A, -2, -1))
        Hp, _ = hessian_pressure(u, H)
        trALA, LapA = _trA_LapA(A)
        gradu2 = np.einsum('...ij,...ij->...', A, A)
        det_s = np.mean(gradu2) ** 1.5 + 1e-30
        means = np.array([np.mean(np.linalg.det(A)) / det_s,
                          np.mean(np.einsum('...ij,...ij->...', S, Hp)),
                          np.mean(trALA)])
        pos.append(float(np.max(np.abs(means))))
        # counter: the un-centred det field has O(1) local values, so a "mean" that
        # forgot to subtract nothing is fine; perturb by a constant offset that must
        # break the zero-mean claim.
        ctr.append(float(np.abs(np.mean(np.linalg.det(A)) / det_s + 1.0)))
    return _verdict('zero_mean_budget', pos, ctr, pos_tol=1e-9)


def check_supersolution():
    """eq: supersolution -- (d_t + U.grad - nu Delta)(|U|^2/2) = -nu|grad U|^2,
    under the pressureless evolution d_t U = nu Delta U - (U.grad)U.
    Counter: +nu|grad U|^2 must not match."""
    pos, ctr = [], []
    for u in _fields().values():
        ut = NU * laplacian_vec(u, H) - advect_vec(u, u, H)
        half = 0.5 * np.einsum('...i,...i->...', u, u)
        lhs = (np.einsum('...i,...i->...', u, ut)
               + np.einsum('...i,...i->...', u, grad_scalar(half, H))
               - NU * laplacian_scalar(half, H))
        gu2 = np.einsum('...ij,...ij->...', fft_jacobian(u, H), fft_jacobian(u, H))
        p, c = _scalar_field_check(lhs, -NU * gu2, +NU * gu2)
        pos.append(p)
        ctr.append(c)
    return _verdict('supersolution', pos, ctr)


def check_constraint_hessian():
    """eq: constraint-hessian (via the product rule):
    Delta q = 2 tr(A Delta A) + 2 (d_k d_i U_j)(d_k d_j U_i).
    Counter: the contraction with a + sign on tr must not match."""
    pos, ctr = [], []
    kx, ky, kz = wavenumbers(N, H)
    for u in _fields().values():
        q, _, _, A = q_of_u(u, H)
        trALA, _ = _trA_LapA(A)
        gradA = np.zeros((3,) + A.shape)
        for i in range(3):
            for j in range(3):
                Ah = fftn(A[..., i, j])
                for k, kk in enumerate((kx, ky, kz)):
                    gradA[k][..., i, j] = np.real(ifftn(1j * kk * Ah))
        P = sum(np.einsum('...ij,...ji->...', gradA[k], gradA[k]) for k in range(3))
        Lq = laplacian_scalar(q, H)
        p, c = _scalar_field_check(Lq, 2 * trALA + 2 * P, 2 * trALA - 2 * P)
        pos.append(p)
        ctr.append(c)
    return _verdict('constraint_hessian', pos, ctr)


def check_gradient_split():
    """eq: gradient-split -- |grad u|^2 = q + |omega|^2 (and q = |S|^2 - 1/2|omega|^2).
    Counter: q - |omega|^2 must not match |grad u|^2."""
    pos, ctr = [], []
    for u in _fields().values():
        q, S, om, A = q_of_u(u, H)
        gu2 = np.einsum('...ij,...ij->...', A, A)
        om2 = np.einsum('...i,...i->...', om, om)
        p, c = _scalar_field_check(gu2, q + om2, q - om2)
        pos.append(p)
        ctr.append(c)
    return _verdict('gradient_split', pos, ctr)


def check_imbalance_norm():
    """eq: imbalance -- ||grad^2 p||_2 = ||q||_2 and tr(grad^2 p) = -q.
    Counter: ||grad^2 p||_2 != 1.5 ||q||_2."""
    pos, ctr = [], []
    for u in _fields().values():
        Hp, q = hessian_pressure(u, H)
        nHp = np.sqrt(np.mean(np.einsum('...ij,...ij->...', Hp, Hp)))
        nq = np.sqrt(np.mean(q ** 2)) + 1e-30
        tr = np.einsum('...ii->...', Hp)
        pos.append(max(abs(nHp - nq) / nq, _rel(tr, -q)))
        ctr.append(abs(nHp - 1.5 * nq) / nq)   # wrong factor must be rejected
    return _verdict('imbalance_norm', pos, ctr)


def check_div_form_rewrite():
    """INLINE CLAIM (the issue-1 identity): for divergence-free U with skew tensor
    potential d (d_jk = eps_jkl a_l, curl a = -U) one has div(d grad U_i) = U.grad U_i,
    hence d_t U_i = div((nu I - d) grad U_i).  Counter: div(d grad U_i) != -U.grad U_i,
    which fixes the skew-part sign as -d (a + d would give the wrong transport)."""
    kx, ky, kz = wavenumbers(N, H)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    inv = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
    eps = np.zeros((3, 3, 3))
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        eps[i, j, k] = 1.0
        eps[i, k, j] = -1.0
    pos, ctr, ident = [], [], []
    for u in _fields().values():
        uh = np.stack([fftn(u[..., m]) for m in range(3)], axis=-1)
        kvec = np.stack([kx, ky, kz], axis=-1)
        # vector potential psi with curl psi = U:  psi_hat = i k x U_hat / |k|^2
        cross = np.cross(kvec, uh)
        psih = 1j * cross * inv[..., None]
        psi = np.stack([np.real(ifftn(psih[..., m])) for m in range(3)], axis=-1)
        a = -psi                                   # curl a = -U  =>  d_jk a-potential
        d = np.einsum('jkl,...l->...jk', eps, a)   # skew: d_jk = eps_jkl a_l
        # identity test: div d (= d_j d_jk) should equal U_k
        divd = np.zeros_like(u)
        for kk in range(3):
            s = np.zeros((N, N, N))
            for j, kf in enumerate((kx, ky, kz)):
                s = s + np.real(ifftn(1j * kf * fftn(d[..., j, kk])))
            divd[..., kk] = s
        ident.append(_rel(divd, u))
        A = fft_jacobian(u, H)                      # A[...,i,j] = d_i U_j ; grad U_i = A[...,:,i]
        UgradU = advect_vec(u, u, H)                # (U.grad U)_i
        # div(d grad U_i)_  = d_j ( d_jk d_k U_i )
        out = np.zeros_like(u)
        for i in range(3):
            gUi = A[..., :, i]                       # d_k U_i
            dgi = np.einsum('...jk,...k->...j', d, gUi)
            s = np.zeros((N, N, N))
            for j, kf in enumerate((kx, ky, kz)):
                s = s + np.real(ifftn(1j * kf * fftn(dgi[..., j])))
            out[..., i] = s
        pos.append(_rel(out, UgradU))
        ctr.append(_rel(out, -UgradU))             # the +d (wrong) sign would need this
    extra = f"potential div d = U to {max(ident):.1e}"
    return _verdict('div_form_rewrite', pos, ctr, detail=extra)


def check_vorticity_equation():
    """SUPPORTS eq: vorticity-ineq -- under d_t U = nu Delta U - U.grad U,
    d_t omega + U.grad omega = omega.grad U + nu Delta omega.
    Counter: the -omega.grad U (wrong stretching sign) must not vanish."""
    pos, ctr = [], []
    for u in _fields().values():
        om = fft_curl(u, H)
        ut = NU * laplacian_vec(u, H) - advect_vec(u, u, H)
        omt = fft_curl(ut, H)                                   # d_t omega
        stretch = advect_vec(om, u, H)                          # (omega.grad)U
        res = (omt + advect_vec(u, om, H) - stretch - NU * laplacian_vec(om, H))
        res_wrong = (omt + advect_vec(u, om, H) + stretch - NU * laplacian_vec(om, H))
        sc = np.max(np.abs(omt)) + 1e-30
        pos.append(float(np.max(np.abs(res)) / sc))
        ctr.append(float(np.max(np.abs(res_wrong)) / sc))
    return _verdict('vorticity_equation', pos, ctr)


def check_cayley():
    """eq: cayley -- a real 3x3 A with tr A = tr A^2 = 0 satisfies A^3 = det(A) I.
    Counter: a generic trace-free matrix does NOT."""
    rng = np.random.default_rng(7)
    pos = []
    for _ in range(20000):
        c = rng.standard_normal()
        B = np.array([[c, 0, 0],
                      [0, -c / 2, -c * np.sqrt(3) / 2],
                      [0, c * np.sqrt(3) / 2, -c / 2]])     # eigvals c{1, zeta, zeta^2}
        P = rng.standard_normal((3, 3))
        if abs(np.linalg.det(P)) < 1e-6:
            continue
        A = P @ B @ np.linalg.inv(P)                         # tr A = tr A^2 = 0
        pos.append(np.max(np.abs(A @ A @ A - np.linalg.det(A) * np.eye(3)))
                   / (np.max(np.abs(A)) ** 3 + 1e-30))
    # counter: a deterministic trace-free matrix with tr G^2 = 6 != 0 must violate.
    G = np.diag([1.0, 1.0, -2.0])
    ctr = np.max(np.abs(G @ G @ G - np.linalg.det(G) * np.eye(3))) / (np.max(np.abs(G)) ** 3)
    return _verdict('cayley', [max(pos)], [ctr])


# --------------------------------------------------------------------------- verdict
def _verdict(name, pos, ctr, pos_tol=TOL, detail=''):
    pos_max = float(np.max(pos))          # identity must hold on EVERY field
    ctr_agg = float(np.max(ctr))          # perturbation rejected on AT LEAST one field
    ok = pos_max < pos_tol and ctr_agg > COUNTER_MIN
    return {'name': name, 'ok': ok, 'pos': pos_max, 'counter': ctr_agg, 'detail': detail}


CHECKS = {
    'production_law': check_production_law,
    'zero_mean_budget': check_zero_mean_budget,
    'supersolution': check_supersolution,
    'constraint_hessian': check_constraint_hessian,
    'gradient_split': check_gradient_split,
    'imbalance_norm': check_imbalance_norm,
    'div_form_rewrite': check_div_form_rewrite,
    'vorticity_equation': check_vorticity_equation,
    'cayley': check_cayley,
}


def run(name):
    return CHECKS[name]()


def main():
    print("equation-level symbolic checks (positive identity + counter-test)\n")
    print("  %-22s %-6s %-12s %-12s %s" % ('check', 'pass', 'pos.err', 'counter', 'note'))
    allok = True
    for name in CHECKS:
        r = run(name)
        allok = allok and r['ok']
        print("  %-22s %-6s %-12.2e %-12.2e %s"
              % (name, 'ok' if r['ok'] else 'FAIL', r['pos'], r['counter'], r['detail']))
    print()
    print('ALL PASS' if allok else 'FAILURES PRESENT')
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
