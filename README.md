# Configuration-Space Curvature and the Navier–Stokes Singular Set

A reading of the three-dimensional Navier–Stokes singular set through the curvature
of the configuration group of an ideal fluid.

**Author:** Alejandro José Soto Franco · **Paper:** [`paper/main.pdf`](paper/main.pdf)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20600978.svg)](https://doi.org/10.5281/zenodo.20600978)

## Summary

At a singular point of a suitable weak solution of the three-dimensional
Navier–Stokes equations, the partial-regularity theory of Caffarelli, Kohn and
Nirenberg leaves two possibilities: either a configuration-space curvature
concentrates, or the strain energy and the enstrophy come into balance. The paper
controls the first possibility coercively and constrains the second, through a
geometric identity:

- The curvature of the `L²` metric on the group `SDiff(T³)` of volume-preserving
  diffeomorphisms is the **pressure Hessian**, the Gauss curvature of `SDiff(T³)`
  inside the flat group `Diff(T³)`, whose second fundamental form is the pressure
  gradient.
- Its `L²` size is the **strain–enstrophy imbalance** `q = |S|² − ½|ω|²`. The
  configuration-space curvature is therefore the pressure Hessian of the velocity-gradient
  dynamics.
- The imbalance obeys a transport law with no inertial and no curvature source.
  In the scale-critical `L³` class (the local form of the
  Escauriaza–Seregin–Šverák condition) the balanced possibility forces a bounded
  ancient pressureless flow. A Liouville theorem for divergence-free drifts
  excludes that flow. Every such singular point, the Type-I points among them, is
  therefore curvature-concentrating.
- Beyond the critical class the balanced regime is supercritical. There the
  curvature is the structure of the nonlinearity that the energy identity does not
  see, the structure an energy-preserving averaged equation discards, which locates
  where regularity is to be decided.

## Repository layout

```
paper/      LaTeX sources, bundled KMS styles, and the compiled main.pdf
verify/     symbolic and spectral verification of every algebraic identity
Makefile    build and verification targets
```

## Building the paper

The paper is self-contained; the KMS styles are bundled in `paper/`.

```bash
make pdf          # -> paper/main.pdf
```

A recent TeX Live with `latexmk` and `biber` is required.

## Reproducing the verification

The verification has three layers, all driven by `uv` (no manual environment setup):

```bash
make verify       # runs the suites below and the coverage check
```

- `verify/cf2_curvature_pressure_channel_verify.py` checks the curvature identities:
  the flat ambient part vanishes, the curvature operator is the Leray-projected
  pressure Hessian, the Gauss equation, and `‖∇²p‖₂ = ‖q‖₂`.
- `verify/cf3_imbalance_production_verify.py` checks the imbalance-production law,
  its zero-mean balance, the energy supersolution, and the constraint reformulation.
- `verify/eqn_lint.py` is an **equation-coverage check**: every displayed identity in
  the paper (`\label{eq: …}`) is classified in `verify/equation_registry.toml`, and
  every algebraic one is bound to a symbolic check in `verify/equation_checks.py`
  that passes together with a **sign-flip counter-test** (the perturbed identity must
  be rejected, which catches a vacuously-passing check). The check fails on any
  unregistered or stale equation, or any algebraic identity without a passing check.

Run the coverage check on its own with `make eqn-lint`, or the symbolic suites with
`make sympy`. The checks use a Fourier pseudo-spectral discretisation on several
divergence-free families. Agreement is therefore not an artefact of a single symmetry.

## Citation

The paper is a preprint, archived on Zenodo together with the verification code at
[`10.5281/zenodo.20600978`](https://doi.org/10.5281/zenodo.20600978). The concept DOI
always resolves to the latest version.

```bibtex
@misc{sotofranco2026curvature,
  author       = {Soto Franco, Alejandro Jos\'e},
  title        = {Configuration-Space Curvature and the Navier--Stokes Singular Set},
  year         = {2026},
  howpublished = {Preprint, Zenodo},
  doi          = {10.5281/zenodo.20600978},
  url          = {https://doi.org/10.5281/zenodo.20600978},
}
```

## Licence

The paper under `paper/` is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The verification code
under `verify/` is licensed under the MIT licence; see [`LICENSE`](LICENSE).
