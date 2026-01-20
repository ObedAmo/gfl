# gfl - Generalized Fused Lasso Solvers

`gfl` provides solvers for the generalized fused lasso (GFL), a penalized least-squares problem that encourages equality between selected pairs of parameters. The package implements coordinate descent methods following [Ohishi et al. (2021)](https://www.tandfonline.com/doi/full/10.1080/03610926.2021.1931888).

⚠️ Project status

This repository is under active development. The public API may change, and the implementation is intended for research and experimental use.

## Problem formulation

We consider the optimization problem of the form

\[
\min_{\theta \in \mathbb{R}^m}
\;\frac{1}{2}\sum_{i=1}^n (y_i - \theta_{g_i})^2
\;+\;
\lambda \sum_{(u,v)\in \mathcal{E}} w_{uv}\,|\theta_u - \theta_v|,
\]

where:

- \(y_i\) are observed responses,
- \(g_i \in \{1,\dots,m\}\) assigns observation \(i\) to a group,
- \(\theta_j\) is a scalar parameter associated with group \(j\),
- \(\mathcal{E}\) is a set of index pairs defining fusion constraints,
- \(w_{uv} > 0\) are fusion weights,
- \(\lambda \ge 0\) controls the strength of fusion.

This formulation encourages **piecewise-constant group parameters** by shrinking differences between selected pairs \((u,v)\). It generalizes the classical fused lasso by allowing arbitrary weighted fusion pairs, without assuming any particular graph or geometric structure unless imposed by the user.

This formulation is equivalent to Equation (3) in  
**Ohishi et al. (2021), _Coordinate optimization for generalized fused Lasso_**.

## Scope

The goal of this package is to provide:

- a clear reference implementation of the coordinate descent algorithms for GFL,
- a stable core API suitable for research and methodological development,
- optional high-performance backends (e.g., C++ extensions)

Application-specific models (spatial, graph-based, robust variants, etc.) will be built on top of this core solver.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
