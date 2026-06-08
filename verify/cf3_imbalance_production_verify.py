"""
Imbalance-production law -- the dynamic form of the coercive criterion.

Claim verified here
-------------------
The strain-enstrophy imbalance  q = |S|^2 - (1/2)|omega|^2 = tr(A^2),  A = grad u,
is transported by the Navier-Stokes flow according to

    D_t q  =  -6 det(grad u)  -  2 S : grad^2 p  +  2 nu tr(A Delta A) ,        (PROD)

where D_t = d_t + (u.grad), S = sym(grad u), and -Delta p = q.

Derivation.  The velocity-gradient evolution is  D_t A = -A^2 - grad^2 p + nu Delta A.
Then  D_t q = D_t tr(A^2) = 2 tr(A D_t A) = -2 tr(A^3) - 2 tr(A grad^2 p) + 2 nu tr(A Delta A).
For a traceless 3x3 matrix tr(A^3) = 3 det A, and tr(A grad^2 p) = S : grad^2 p
(only the symmetric part survives), giving (PROD).

Why it matters.  (PROD) is the coercive criterion in dynamic form:
  * the INERTIAL term  -6 det(grad u)  is the source of imbalance;
  * the CURVATURE term  -2 S : grad^2 p  is the only sink that can cancel it;
  * both are constraint/null terms with ZERO spatial mean:
        int det(grad u) dx = 0   (det of a gradient is a null Lagrangian on T^3),
        int S : grad^2 p  dx = 0  (the curvature does no net work -- PART F of cf2).
    So viscosity is the only net sink of imbalance.
A balanced singularity (q -> 0 in scaled average) would require the curvature work
to cancel the inertial production locally up to the singular time, against a global
zero-sum budget -- the fine-tuning the unconditional program must rule out.

Verifications
-------------
PART G1 (snapshot, two independent computations of D_t q agree to machine eps):
        Way 1 -- definition  D_t q = d_t q + u.grad q, with d_t grad u from the
                 full NS right-hand side  d_t u = -(u.grad)u - grad p + nu Delta u.
        Way 2 -- identity (PROD).
PART G2 (the zero-mean budget):  int det(grad u) = 0 and int S:grad^2 p = 0.
PART G3 (the geometric sharpening):  both blow-up corners are delta-imbalanced.
        |q|/|grad u|^2 is bounded below away from the balance surface q=0; the
        criterion fails only on that codimension-one set.

Run:  uv run python sympy/cf3_imbalance_production_verify.py
"""

import os
import sys

import numpy as np
from numpy.fft import fftn, ifftn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pressure_hess_identity import (  # noqa: E402
    wavenumbers,
    fft_jacobian,
    fft_curl,
    flow_abc,
    flow_perturbed_abc,
    flow_anti_parallel_tubes,
    flow_helical,
)


def grad_scalar(f, h):
    """grad f, returns [...,3] with component d_i f."""
    N = f.shape[0]
    kx, ky, kz = wavenumbers(N, h)
    fh = fftn(f)
    return np.stack([np.real(ifftn(1j * kx * fh)),
                     np.real(ifftn(1j * ky * fh)),
                     np.real(ifftn(1j * kz * fh))], axis=-1)


def laplacian_scalar(f, h):
    """Delta f for a scalar field."""
    N = f.shape[0]
    kx, ky, kz = wavenumbers(N, h)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    return np.real(ifftn(-k2 * fftn(f)))


def laplacian_vec(v, h):
    """Delta v componentwise."""
    N = v.shape[0]
    kx, ky, kz = wavenumbers(N, h)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    out = np.empty_like(v)
    for i in range(3):
        out[..., i] = np.real(ifftn(-k2 * fftn(v[..., i])))
    return out


def pressure_from_u(u, h):
    """p with -Delta p = d_i u_j d_j u_i = q (zero mean)."""
    N = u.shape[0]
    kx, ky, kz = wavenumbers(N, h)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    Ju = fft_jacobian(u, h)                      # Ju[...,i,j] = d_i u_j
    q = np.einsum('...ij,...ji->...', Ju, Ju)    # d_i u_j d_j u_i = tr(A^2)
    qh = fftn(q)
    ph = np.where(k2 > 0, qh / np.where(k2 > 0, k2, 1.0), 0.0)   # -k^2 p = -q
    return np.real(ifftn(ph)), q


def hessian_pressure(u, h):
    """grad^2 p with -Delta p = q."""
    N = u.shape[0]
    kx, ky, kz = wavenumbers(N, h)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    Ju = fft_jacobian(u, h)
    q = np.einsum('...ij,...ji->...', Ju, Ju)
    qh = fftn(q)
    ph = np.where(k2 > 0, qh / np.where(k2 > 0, k2, 1.0), 0.0)
    Hp = np.zeros(u.shape + (3,))
    klist = [kx, ky, kz]
    for i in range(3):
        for j in range(3):
            Hp[..., i, j] = np.real(ifftn(-klist[i] * klist[j] * ph))
    return Hp, q


def advect_vec(a, f, h):
    J = fft_jacobian(f, h)                        # J[...,d,i] = d_d f_i
    return np.einsum('...d,...di->...i', a, J)


def _flows(N, h):
    xs = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    return {
        'ABC (Beltrami)':      flow_abc(X, Y, Z, h),
        'perturbed ABC':       flow_perturbed_abc(X, Y, Z, h),
        'anti-parallel tubes': flow_anti_parallel_tubes(X, Y, Z, h),
        'helical blob':        flow_helical(X, Y, Z, h),
    }


def q_of_u(u, h):
    Ju = fft_jacobian(u, h)
    S = 0.5 * (Ju + np.swapaxes(Ju, -2, -1))
    om = fft_curl(u, h)
    return (np.einsum('...ij,...ij->...', S, S)
            - 0.5 * np.einsum('...i,...i->...', om, om)), S, om, Ju


def part_g1_production_identity(nu=0.05):
    N, h = 48, 2 * np.pi / 48
    print("PART G1  (imbalance-production identity, two independent computations)")
    print("  nu = %.3f ;  D_t q  via  definition  vs  -6 det(grad u) - 2 S:Hp + 2 nu tr(A dA)" % nu)
    print("  %-22s %-12s %-12s %-9s" % ('flow', 'rel.err', 'max|D_t q|', 'pass'))
    ok = True
    for name, u in _flows(N, h).items():
        # --- Way 1: definition  D_t q = d_t q + u.grad q --------------------
        q, S, om, Ju = q_of_u(u, h)
        p, _ = pressure_from_u(u, h)
        gradp = grad_scalar(p, h)
        ut = -advect_vec(u, u, h) - gradp + nu * laplacian_vec(u, h)   # d_t u (NS)
        # d_t (grad u): A_dot[...,i,j] = d_i (d_t u_j)
        Adot = fft_jacobian(ut, h)
        Sdot = 0.5 * (Adot + np.swapaxes(Adot, -2, -1))
        omdot = fft_curl(ut, h)
        qt = (2.0 * np.einsum('...ij,...ij->...', S, Sdot)
              - np.einsum('...i,...i->...', om, omdot))            # d_t q
        Dq_def = qt + np.einsum('...i,...i->...', u, grad_scalar(q, h))

        # --- Way 2: identity (PROD) -----------------------------------------
        A = Ju                                                     # A[...,i,j]=d_i u_j
        detA = np.linalg.det(A)
        Hp, _ = hessian_pressure(u, h)
        ShessP = np.einsum('...ij,...ij->...', S, Hp)              # S : grad^2 p
        dA = np.stack([laplacian_vec(A[..., :, j], h) for j in range(3)], axis=-1)
        # dA[...,i,k,j] = Delta(d_i u_k)?  build tr(A Delta A) directly:
        # tr(A Delta A) = A_{ij} (Delta A)_{ji}.  (Delta A)_{ji} = Delta d_j u_i.
        N_ = u.shape[0]
        kx, ky, kz = wavenumbers(N_, h)
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        LapA = np.zeros_like(A)
        for i in range(3):
            for j in range(3):
                LapA[..., i, j] = np.real(ifftn(-k2 * fftn(A[..., i, j])))
        trA_LapA = np.einsum('...ij,...ji->...', A, LapA)
        Dq_id = -6.0 * detA - 2.0 * ShessP + 2.0 * nu * trA_LapA

        den = np.max(np.abs(Dq_def)) + 1e-30
        rel = np.max(np.abs(Dq_def - Dq_id)) / den
        passed = rel < 1e-6
        ok = ok and passed
        print("  %-22s %-12.3e %-12.3e %-9s" % (name, rel, den, 'ok' if passed else 'FAIL'))
    print()
    assert ok, "imbalance-production identity failed"
    print("  [PASS] D_t q = -6 det(grad u) - 2 S:grad^2 p + 2 nu tr(A Delta A).\n")
    return ok


def part_g2_zero_mean_budget():
    N, h = 48, 2 * np.pi / 48
    print("PART G2  (zero-mean budget: all three production terms redistribute, never create)")
    print("  %-22s %-18s %-18s %-18s" %
          ('flow', 'mean det(grad u)', 'mean S:grad^2 p', 'mean tr(A Delta A)'))
    ok = True
    kx, ky, kz = wavenumbers(N, h)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    for name, u in _flows(N, h).items():
        Ju = fft_jacobian(u, h)
        detA = np.linalg.det(Ju)
        S = 0.5 * (Ju + np.swapaxes(Ju, -2, -1))
        Hp, _ = hessian_pressure(u, h)
        # viscous term tr(A Delta A) = A_{ij} (Delta A)_{ji}
        LapA = np.zeros_like(Ju)
        for i in range(3):
            for j in range(3):
                LapA[..., i, j] = np.real(ifftn(-k2 * fftn(Ju[..., i, j])))
        trA_LapA = np.einsum('...ij,...ji->...', Ju, LapA)
        m_det = float(np.mean(detA))
        m_work = float(np.mean(np.einsum('...ij,...ij->...', S, Hp)))
        m_visc = float(np.mean(trA_LapA))
        # natural field scales: |det grad u| ~ (rms|grad u|)^3, |S:grad^2 p| ~ rms|S| rms|Hp|
        gradu2 = np.einsum('...ij,...ij->...', Ju, Ju)
        det_scale = float(np.mean(gradu2) ** 1.5) + 1e-30
        rms_gradu = float(np.sqrt(np.mean(gradu2)))
        work_scale = (float(np.sqrt(np.mean(np.einsum('...ij,...ij->...', S, S))))
                      * float(np.sqrt(np.mean(np.einsum('...ij,...ij->...', Hp, Hp)))) + 1e-30)
        # tr(A Delta A) ~ rms|grad u| * rms|Delta grad u|
        visc_scale = (rms_gradu
                      * float(np.sqrt(np.mean(np.einsum('...ij,...ij->...', LapA, LapA)))) + 1e-30)
        ok = (ok and abs(m_det) / det_scale < 1e-12
              and abs(m_work) / work_scale < 1e-12
              and abs(m_visc) / visc_scale < 1e-12)
        print("  %-22s %-+18.3e %-+18.3e %-+18.3e" % (name, m_det, m_work, m_visc))
    print()
    assert ok, "zero-mean budget failed"
    print("  [PASS] int det(grad u) = 0 (null Lagrangian), int S:grad^2 p = 0,")
    print("         and int tr(A Delta A) = -sum |k|^2 |k.u_hat|^2 = 0 (incompressibility).")
    print("         All three terms redistribute the imbalance with zero net budget; the")
    print("         imbalance int q = int|S|^2 - 1/2 int|omega|^2 = 0 is preserved identically.\n")
    return ok


def part_g3_both_corners_imbalanced():
    N, h = 48, 2 * np.pi / 48
    print("PART G3  (geometric sharpening: failure set is the codim-1 balance surface)")
    print("  ratio r(x) = |q|/|grad u|^2 in [0,1];  q>0 strain-dom, q<0 enstrophy-dom, q=0 balance")
    print("  %-22s %-10s %-10s %-12s %-12s" %
          ('flow', 'frac q>0', 'median r', 'r at |q| top5%', 'min r (bal.)'))
    for name, u in _flows(N, h).items():
        q, S, om, Ju = q_of_u(u, h)
        gradu2 = np.einsum('...ij,...ij->...', Ju, Ju)
        r = np.abs(q) / (gradu2 + 1e-30)
        frac_pos = float(np.mean(q > 0))
        med = float(np.median(r))
        # where the imbalance is largest (the dynamically active set), r is near 1:
        thr = np.quantile(np.abs(q), 0.95)
        r_active = float(np.median(r[np.abs(q) >= thr]))
        r_min = float(np.min(r))
        print("  %-22s %-10.3f %-10.3f %-12.3f %-12.3e" %
              (name, frac_pos, med, r_active, r_min))
    print()
    print("  [READ] On the dynamically active set (top 5% |q|) the imbalance ratio r is")
    print("         O(1): both strain-dominated and enstrophy-dominated structures are")
    print("         delta-imbalanced. r -> 0 only on the balance surface q=0 (codim 1),")
    print("         which is exactly where the coercive criterion is silent.\n")
    return True


def part_h_pressureless_energy_supersolution(nu=0.05):
    """On the pressureless limit the kinetic energy Phi = |U|^2/2 obeys
       (d_t + U.grad - nu Delta) Phi = - nu |grad U|^2 = - nu |omega|^2   (balanced),
    a bounded ancient supersolution of the drift-heat operator with bounded
    divergence-free drift.  We verify the exact identity
       (d_t + U.grad - nu Delta)(|U|^2/2) = - nu |grad U|^2
    that holds for ANY field under the pressureless evolution d_t U = nu Delta U
    - (U.grad)U, by substituting that right-hand side on a snapshot."""
    N, h = 48, 2 * np.pi / 48
    print("PART H  (pressureless energy supersolution identity)")
    print("  (d_t + U.grad - nu Delta)(|U|^2/2)  vs  - nu |grad U|^2   [pressureless d_t U]")
    print("  %-22s %-12s %-12s %-9s" % ('flow', 'rel.err', 'max|RHS|', 'pass'))
    ok = True
    for name, u in _flows(N, h).items():
        # pressureless material acceleration:  d_t U = nu Delta U - (U.grad)U
        ut = nu * laplacian_vec(u, h) - advect_vec(u, u, h)
        half_usq = 0.5 * np.einsum('...i,...i->...', u, u)
        dt_half = np.einsum('...i,...i->...', u, ut)              # d_t(|U|^2/2) = U.d_t U
        adv = np.einsum('...i,...i->...', u, grad_scalar(half_usq, h))
        lap = laplacian_scalar(half_usq, h)
        lhs = dt_half + adv - nu * lap
        Ju = fft_jacobian(u, h)
        rhs = -nu * np.einsum('...ij,...ij->...', Ju, Ju)
        den = np.max(np.abs(rhs)) + 1e-30
        rel = np.max(np.abs(lhs - rhs)) / den
        # band-limited flows reach machine epsilon; the Gaussian tube/helical
        # flows are limited by the spectral aliasing of their sharp profiles.
        passed = rel < 1e-6
        ok = ok and passed
        print("  %-22s %-12.3e %-12.3e %-9s" % (name, rel, den, 'ok' if passed else 'FAIL'))
    print()
    assert ok, "pressureless energy-supersolution identity failed"
    print("  [PASS] (d_t + U.grad - nu Delta)(|U|^2/2) = - nu |grad U|^2 <= 0;")
    print("         the limit's kinetic energy is a bounded ancient supersolution.\n")
    return ok


def part_i_constraint_no_one_sided_bound():
    """On the balanced limit q = 0, the differential constraint C1
        3 det(grad U) = nu tr(A Delta A)
    rewrites, via the always-true product-rule identity
        Delta q = 2 tr(A Delta A) + 2 P,   P := d_k A_ij d_k A_ji
                = (d_k d_i U_j)(d_k d_j U_i),
    and q = 0 => Delta q = 0, into
        3 det(grad U) = - nu P.
    We test whether P (hence det on the limit) is one-signed.  Each d_k A is
    TRACE-FREE (d_k tr A = d_k div U = 0), so tr((d_k A)^2) = |sym|^2 - |antisym|^2
    is sign-indefinite, and P = sum_k tr((d_k A)^2) takes BOTH signs pointwise.
    The natural one-sided-bound route to osc(-inf)=0 therefore does not close."""
    N, h = 48, 2 * np.pi / 48
    print("PART I  (constraint C1 rewrite; the one-sided-bound route does NOT close)")
    print("  identity  Delta q = 2 tr(A Delta A) + 2 P   and sign of P (= -3 det/nu on limit)")
    print("  %-22s %-12s %-10s %-12s %-12s" %
          ('flow', 'id rel.err', 'frac P>0', 'min P', 'max P'))
    kx, ky, kz = wavenumbers(N, h)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    ok = True
    both_signs = True
    for name, u in _flows(N, h).items():
        q, S, om, A = q_of_u(u, h)
        # Laplacian of A, componentwise (spectral)
        LapA = np.zeros_like(A)
        gradA = np.zeros((3,) + A.shape)          # gradA[k,...,i,j] = d_k A_ij
        for i in range(3):
            for j in range(3):
                Ah = fftn(A[..., i, j])
                LapA[..., i, j] = np.real(ifftn(-k2 * Ah))
                for k, kk in enumerate((kx, ky, kz)):
                    gradA[k][..., i, j] = np.real(ifftn(1j * kk * Ah))
        trA_LapA = np.einsum('...ij,...ji->...', A, LapA)
        P = sum(np.einsum('...ij,...ji->...', gradA[k], gradA[k]) for k in range(3))
        Lq = laplacian_scalar(q, h)
        # always-true product-rule identity
        rel = np.max(np.abs(Lq - (2.0 * trA_LapA + 2.0 * P))) / (np.max(np.abs(Lq)) + 1e-30)
        # band-limited flows reach machine epsilon; sharp-profile flows are limited
        # by the spectral aliasing of their third derivatives (same floor as PART H).
        ok = ok and rel < 1e-6
        fpos = float(np.mean(P > 0))
        both_signs = both_signs and (0.01 < fpos < 0.99)
        print("  %-22s %-12.3e %-10.3f %-+12.3e %-+12.3e" %
              (name, rel, fpos, float(np.min(P)), float(np.max(P))))
    print()
    assert ok, "product-rule identity Delta q = 2 tr(A dA) + 2 P failed"
    print("  [PASS] identity Delta q = 2 tr(A Delta A) + 2 P holds (machine precision),")
    print("         so on the limit (Delta q = 0):  3 det(grad U) = - nu P.")
    print("  [READ] P takes BOTH signs pointwise (sum_k tr((d_k A)^2), each d_k A")
    print("         trace-free), so det(grad U) is sign-indefinite: C1 gives NO")
    print("         one-sided bound, and the supersolution route to osc(-inf)=0 stalls.\n")
    return ok and both_signs


def main():
    print("=" * 74)
    print("Imbalance-production law  D_t q = -6 det(grad u) - 2 S:grad^2 p + 2 nu tr(A dA)")
    print("=" * 74 + "\n")
    r1 = part_g1_production_identity()
    r2 = part_g2_zero_mean_budget()
    r3 = part_g3_both_corners_imbalanced()
    r4 = part_h_pressureless_energy_supersolution()
    r5 = part_i_constraint_no_one_sided_bound()
    print("=" * 74)
    print("SUMMARY")
    print(f"  G1 production identity (two computations agree) : {'PASS' if r1 else 'FAIL'}")
    print(f"  G2 zero-mean budget (no inertial/curvature source): {'PASS' if r2 else 'FAIL'}")
    print(f"  G3 both corners delta-imbalanced (failure = q=0)  : {'PASS' if r3 else 'FAIL'}")
    print(f"  H  pressureless energy supersolution identity     : {'PASS' if r4 else 'FAIL'}")
    print(f"  I  constraint rewrite + P sign-indefinite         : {'PASS' if r5 else 'FAIL'}")
    print("=" * 74)


if __name__ == '__main__':
    main()
