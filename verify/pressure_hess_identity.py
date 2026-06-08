"""
Pressure-Hessian identity for the dyadic strain-energy evolution.

Goal: derive and verify a clean expression for
   I(t)  :=  int_{T^3}  xi^T S nabla^2 p  xi  dx,
the pressure-Hessian term in the distilled  D_t(xi^T S^2 xi)  equation:
   D_t(xi^T S^2 xi)  =  -2 alpha (xi^T S^2 xi)
                       -  2 xi^T S nabla^2 p xi
                       +  2 nu xi^T S Delta S xi.

Background
----------
Pressure Poisson:  -Delta p  =  (partial_i u_j)(partial_j u_i)  =  |S|^2 - |Omega|^2  =  |S|^2 - (1/2)|omega|^2.
Let  q := |S|^2 - (1/2)|omega|^2.  Then  p = (-Delta)^{-1} q,  and
   nabla^2 p  =  -R q,
where R is the matrix Riesz operator  R_{ij} = partial_i partial_j (-Delta)^{-1},
symbol  eta_i eta_j / |eta|^2.

So pointwise:
   xi^T S nabla^2 p  xi (x)  =  - xi(x)^T S(x) [R q] (x)  xi(x),
which is NONLOCAL: depends on q everywhere.

CLEAN IDENTITIES:

(I1) The "naive" trace integral vanishes:
        int S : nabla^2 p  dx  =  int S_{ij} partial_i partial_j p  dx
                                =  int p  partial_i partial_j S_{ij}  dx   (IBP)
                                =  int p  Delta(partial_j u_j)  dx          (div S = (1/2) Delta u)
                                =  0  by incompressibility.

(I2) The xi-projected integral does NOT vanish in general.
     I(t)  =  int [S xi (x) xi]_{ij} (nabla^2 p)_{ij}  dx
            =  -int [S xi (x) xi]_{ij}  R_{ij} q  dx
     and by Parseval:
            =  -int (eta_i eta_j / |eta|^2) hat[(S xi)_i xi_j]^* hat q d eta / V.

(I3) Pointwise Riesz bound (Calderon-Zygmund):
        |nabla^2 p|(x)  <=  C (M(|S|^2)(x) + M(|omega|^2)(x))
     where M is the Hardy-Littlewood maximal function.

NUMERICAL TEST: compute I(t) on 4 reference flows; report sign + magnitude.
The empirical sign tells us whether the pressure-Hessian DRIVES or RETARDS
the monotone growth of T_xiS2.
"""

import numpy as np
from numpy.fft import fftn, ifftn, fftfreq


def wavenumbers(N, h):
    k = 2 * np.pi * fftfreq(N, d=h)
    return np.meshgrid(k, k, k, indexing='ij')


def fft_curl(v, h):
    N = v.shape[0]
    kx, ky, kz = wavenumbers(N, h)
    v_hat = [fftn(v[..., i]) for i in range(3)]
    return np.stack([
        np.real(ifftn(1j * (ky * v_hat[2] - kz * v_hat[1]))),
        np.real(ifftn(1j * (kz * v_hat[0] - kx * v_hat[2]))),
        np.real(ifftn(1j * (kx * v_hat[1] - ky * v_hat[0]))),
    ], axis=-1)


def fft_jacobian(v, h):
    N = v.shape[0]
    kx, ky, kz = wavenumbers(N, h)
    J = np.zeros(v.shape + (3,))
    for i in range(3):
        f_hat = fftn(v[..., i])
        J[..., 0, i] = np.real(ifftn(1j * kx * f_hat))
        J[..., 1, i] = np.real(ifftn(1j * ky * f_hat))
        J[..., 2, i] = np.real(ifftn(1j * kz * f_hat))
    return J


def biot_savart_u(omega, h):
    N = omega.shape[0]
    kx, ky, kz = wavenumbers(N, h)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2[0, 0, 0] = 1.0
    om_hat = [fftn(omega[..., i]) for i in range(3)]
    u_hat = [
        1j * (ky * om_hat[2] - kz * om_hat[1]) / k2,
        1j * (kz * om_hat[0] - kx * om_hat[2]) / k2,
        1j * (kx * om_hat[1] - ky * om_hat[0]) / k2,
    ]
    for h_arr in u_hat:
        h_arr[0, 0, 0] = 0.0
    return np.stack([np.real(ifftn(h_arr)) for h_arr in u_hat], axis=-1)


def pressure_hess_full(u, h):
    """Returns: p, q = (|S|^2 - |omega|^2/2), nabla^2 p."""
    N = u.shape[0]
    kx, ky, kz = wavenumbers(N, h)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2_safe = np.where(k2 > 0, k2, 1.0)
    Ju = fft_jacobian(u, h)
    S = 0.5 * (Ju + np.swapaxes(Ju, -2, -1))
    om = fft_curl(u, h)
    om_mag_sq = np.sum(om ** 2, axis=-1)
    S_sq = np.einsum('...ij,...ij->...', S, S)
    q = S_sq - 0.5 * om_mag_sq
    q_hat = fftn(q)
    p_hat = np.where(k2 > 0, q_hat / k2_safe, 0.0)
    # nabla^2 p
    Hp = np.zeros(u.shape + (3,))
    for i, ki in enumerate([kx, ky, kz]):
        for j, kj in enumerate([kx, ky, kz]):
            Hp[..., i, j] = np.real(ifftn(-ki * kj * p_hat))
    return Hp, q, S, om


def flow_abc(X, Y, Z, h):
    return np.stack([np.sin(Z) + np.cos(Y),
                     np.sin(X) + np.cos(Z),
                     np.sin(Y) + np.cos(X)], axis=-1)


def flow_perturbed_abc(X, Y, Z, h, amp=0.5):
    u_abc = flow_abc(X, Y, Z, h)
    A = np.stack([amp * np.sin(2*X) * np.cos(Y) * np.cos(Z),
                  amp * np.cos(X) * np.sin(2*Z) * np.cos(Y),
                  amp * np.sin(2*Y) * np.cos(X) * np.cos(Z)], axis=-1)
    return u_abc + fft_curl(A, h)


def flow_anti_parallel_tubes(X, Y, Z, h, R=0.4):
    yc = np.pi
    z1, z2 = np.pi - 0.7, np.pi + 0.7
    om_x = np.exp(-((Y - yc) ** 2 + (Z - z1) ** 2) / R ** 2) \
         - np.exp(-((Y - yc) ** 2 + (Z - z2) ** 2) / R ** 2)
    om = np.stack([om_x, np.zeros_like(X), np.zeros_like(X)], axis=-1)
    om -= np.mean(om, axis=(0, 1, 2), keepdims=True)
    return biot_savart_u(om, h)


def flow_helical(X, Y, Z, h, R=0.5):
    rho_sq = (X - np.pi) ** 2 + (Y - np.pi) ** 2
    f = np.exp(-rho_sq / R ** 2)
    A_twist = np.stack([np.zeros_like(X), np.zeros_like(X),
                        0.4 * np.sin(Z) * f], axis=-1)
    om = np.stack([np.zeros_like(X), np.zeros_like(X), f], axis=-1)
    om = om + fft_curl(A_twist, h)
    om -= np.mean(om, axis=(0, 1, 2), keepdims=True)
    return biot_savart_u(om, h)


def analyse(u, h, name):
    Hp, q, S, om = pressure_hess_full(u, h)
    om_mag = np.linalg.norm(om, axis=-1)
    eps = 1e-12
    xi = om / (om_mag[..., None] + eps)

    # (I1) trace: int S : nabla^2 p dx should be 0.
    trace_integrand = np.einsum('...ij,...ij->...', S, Hp)
    I_trace = float(np.sum(trace_integrand) * h ** 3)

    # I = int xi^T S nabla^2 p xi dx
    # = int (S xi)_a (nabla^2 p)_{ab} xi_b dx
    Sxi = np.einsum('...ab,...b->...a', S, xi)
    Sxi_Hp_xi = np.einsum('...a,...ab,...b->...', Sxi, Hp, xi)
    I_full = float(np.sum(Sxi_Hp_xi) * h ** 3)

    # Mean and sign distribution
    mask = om_mag > 0.1 * om_mag.max()
    mean = float(np.mean(Sxi_Hp_xi[mask]))
    frac_pos = float(np.mean(Sxi_Hp_xi[mask] > 0))
    abs_max = float(np.max(np.abs(Sxi_Hp_xi[mask])))

    # Statistics of q
    q_pos = float(np.mean(q > 0))                       # fraction strain-dominated
    q_mean = float(np.mean(q))

    # Riesz bound: ||nabla^2 p||_L2 <= C ||q||_L2 (Calderón-Zygmund: norm 1).
    Hp_L2 = float(np.sqrt(np.sum(Hp ** 2) * h ** 3))
    q_L2 = float(np.sqrt(np.sum(q ** 2) * h ** 3))
    S_L2 = float(np.sqrt(np.sum(S ** 2) * h ** 3))

    print(f"### {name.strip()}")
    print(f"  q = |S|^2 - |omega|^2/2:")
    print(f"    frac q > 0 (strain-dominated): {q_pos:.4f}")
    print(f"    mean q                       : {q_mean:+.4e}")
    print(f"    ||q||_L2                     : {q_L2:.4e}")
    print(f"  ||nabla^2 p||_L2               : {Hp_L2:.4e}")
    print(f"  ||S||_L2                       : {S_L2:.4e}")
    print(f"  Riesz check: ||nabla^2 p||/||q|| = {Hp_L2/max(q_L2,1e-30):.4f}  (theoretically <= 1)")
    print()
    print(f"  Trace integral  int S:nabla^2 p dx  =  {I_trace:+.4e}   (should be ~0 by IBP)")
    print(f"  Dyadic integral I = int xi^T S nabla^2 p xi dx  =  {I_full:+.4e}")
    print(f"  pointwise statistics on |omega| > 0.1 max:")
    print(f"    mean(xi^T S nabla^2 p xi):  {mean:+.4e}")
    print(f"    frac > 0:                    {frac_pos:.4f}")
    print(f"    max |.|:                     {abs_max:.4e}")
    print()
    return I_full


def main():
    N = 96
    L = 2 * np.pi
    h = L / N
    coords = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')

    flows = {
        'ABC(1,1,1)         ': flow_abc(X, Y, Z, h),
        'perturbed ABC      ': flow_perturbed_abc(X, Y, Z, h),
        'anti-parallel tubes': flow_anti_parallel_tubes(X, Y, Z, h),
        'helical column     ': flow_helical(X, Y, Z, h),
    }

    print("=" * 84)
    print(f"Pressure-Hessian dyadic identity:  I = int xi^T S nabla^2 p xi dx")
    print("=" * 84)
    print()

    results = {}
    for name, u in flows.items():
        results[name] = analyse(u, h, name)

    print("=" * 84)
    print("CROSS-FLOW SUMMARY of I(t) at t=0:")
    print("=" * 84)
    for name, val in results.items():
        contrib_to_dT_dt = -2 * val   # since the term in D_t is  -2 xi^T S nabla^2 p xi
        print(f"  {name:21s}  I = {val:+12.4e},  contribution to dT/dt  =  {contrib_to_dT_dt:+12.4e}")
    print()
    print("Recall: total dT/dt = -2 int alpha J dx  -  2 I  +  2 nu int xi^T S Delta S xi.")
    print("If I < 0  =>  the pressure-Hessian term  -2 I > 0  DRIVES growth (Perelman-helpful).")
    print("If I > 0  =>  the pressure-Hessian opposes growth (combines with alpha-depletion).")


if __name__ == "__main__":
    main()
