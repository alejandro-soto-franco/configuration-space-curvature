SHELL := /bin/bash
.PHONY: pdf verify sympy eqn-lint clean

# Build the paper. Self-contained: the KMS styles are bundled in paper/.
pdf:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# Full verification of the paper's identities.
verify: sympy eqn-lint

# Symbolic and spectral identity suites (Appendix A).
sympy:
	uv run --with numpy --with sympy python verify/cf2_curvature_pressure_channel_verify.py
	uv run --with numpy python verify/cf3_imbalance_production_verify.py

# Equation-coverage gate: every displayed identity in the paper is classified in
# verify/equation_registry.toml, and every algebraic one is bound to a symbolic check
# that passes together with a sign-flip counter-test (verify/equation_checks.py).
eqn-lint:
	uv run --with numpy --with sympy python verify/eqn_lint.py paper

clean:
	cd paper && latexmk -c main.tex || true
	rm -rf verify/__pycache__
