# Scientific Computing Portfolio: A Differentiable Master Function

This is an independent portfolio project for scientific-computing and
computational-scientist roles. Its purpose is to demonstrate how a complex
mathematical function can be translated into readable, testable, differentiable,
and numerically reliable software.

The first case study is a Master Function adapted from the author's earlier
research. It is used here as a challenging computational kernel—not as an
attempt to reproduce the paper or rebuild the complete original algorithm.
Given a candidate plane normal and five parameterized lines, the function
constructs a plane, computes intersections, fits a planar conic, and evaluates
two nonlinear constraints.

The portfolio emphasis is on numerical software engineering: mathematical
decomposition, automatic differentiation, validation, conditioning, failure
handling, reproducibility, profiling, and communication of results.

## Mathematical pipeline

The current implementation composes the following numerical stages:

```text
candidate normal
  -> normalize the normal vector
  -> construct an orthonormal frame for the plane
  -> intersect five lines of sight with the plane
  -> express the intersections in 2D plane coordinates
  -> fit ax² + by² + cxy + dx + ey + 1 = 0
  -> evaluate the two focal constraints
  -> return F(w) in R²
```

The public entry point is:

```python
master_function(normal_vector, position_matrix, direction_matrix)
```

where the normal has shape `(3,)`, and the position and direction matrices have
shape `(3, 5)` with one observation per column.

## Current status

Milestone 1 was completed on **2026-08-05**. The first end-to-end computational
kernel is implemented in `src/core.py` with these stages:

- `normalize_normal_vector`
- `create_orthonormal_frame`
- `intersect_LOS_with_plane`
- `fit_conic`
- `compute_free_terms`
- `master_function`

The unit-test suite currently verifies normalization, frame orthonormality,
line-plane intersections, conic fitting, the focal-constraint formula, and an
end-to-end sampled ellipse whose Master Function value is zero.

Run the tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

Current result: **8 tests pass**.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Double precision is preferred for the numerical experiments:

```python
dtype = torch.float64
```

## Repository layout

```text
src/
  __init__.py       Public package API
  core.py           Master Function and its numerical stages
tests/
  test_core.py      Unit and end-to-end tests
examples/
  basic_usage.py    Minimal runnable example (scheduled for Milestone 2)
requirements.txt    Runtime dependencies
```

## Next milestone — 2026-08-08

The next project session will turn the tested implementation into a minimal,
reproducible demonstration:

1. Replace `examples/basic_usage.py` with a valid three-input Master Function
   example based on the sampled ellipse used in the end-to-end test.
2. Document the scientific problem, exact command, and expected output.
3. Record an initial runtime baseline for function evaluation and, if time
   permits, Jacobian evaluation.
4. Add one degenerate or failure case, such as a line parallel to the candidate
   orbital plane.
5. Run the example and tests from a clean environment, then create a focused Git
   commit.

The current example still targets an earlier one-argument prototype and is a
known item for this milestone.

## Project roadmap

### Milestone 2 — Minimal runnable example (2026-08-08)

Deliver a clean command-line example that a new reader can reproduce from the
README, together with expected output and a first performance baseline.

### Milestone 3 — Differentiation and numerical experiment (2026-08-15)

- Compute the Jacobian of `F` using PyTorch automatic differentiation.
- Validate it against finite differences or an analytic special case.
- Study sensitivity or conditioning as the input approaches a degenerate case.
- Fix random seeds and record the complete environment.
- Produce at least one explained result figure or table.
- Document a numerical failure or degenerate case.

### Milestone 4 — Portfolio presentation and benchmark (2026-08-22)

- Present the computational problem, implementation, validation, and
  reproduction steps without requiring readers to know the original research.
- Compare the PyTorch implementation with a meaningful baseline, such as the
  earlier implementation, a NumPy version, or finite-difference Jacobian
  evaluation.
- Add a diagram or result visualization.
- Link the code and resume description; cite the earlier paper only as the
  provenance of the case study.

### Milestone 5 — Interview-ready demonstration (2026-08-29)

Prepare and record a ten-minute project explanation covering the problem,
mathematical pipeline, design decisions, numerical validation, performance,
limitations, and next steps.

## Definition of portfolio-ready

For job-search purposes, this project is complete when it provides:

- a correct, well-decomposed implementation of a nontrivial mathematical
  function with clearly documented inputs, outputs, and assumptions;
- a tested Jacobian computed with PyTorch automatic differentiation;
- comparison against an independent reference such as finite differences,
  analytic special cases, or a second implementation;
- unit, integration, degeneracy, numerical-accuracy, and dtype tests;
- a reproducible example that runs from a fresh environment;
- a benchmark covering runtime and, where useful, numerical accuracy;
- at least one interpretable figure or table showing behavior, sensitivity, or
  performance;
- clear documentation of the mathematical problem, implementation choices,
  limitations, and results;
- a concise project page, resume bullets, and a ten-minute technical talk.

The project is intentionally not scoped as a reproduction of the earlier paper.
Implementing its complete projective-plane subdivision system, interval
arithmetic, or domain-specific decision rules is outside the core portfolio
scope. A polished computational kernel, validated derivatives, numerical study,
benchmark, and clear technical narrative form a coherent flagship project on
their own. A second, unrelated numerical kernel can be added later only if it
strengthens the general scientific-computing story.
