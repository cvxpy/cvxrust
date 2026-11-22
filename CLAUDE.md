# cvxrust Development Guide

cvxrust is a Disciplined Convex Programming library for Rust. It verifies
convexity, canonicalizes expressions, and solves problems through Clarabel.

## Commands

Use a Rust toolchain compatible with `Cargo.toml`, with `rustfmt` and `clippy`
installed. Run commands from the repository root:

```sh
make          # List all available commands
make build    # Build the library
make test     # Run tests, including doctests
make fmt      # Format Rust sources
make lint     # Check formatting and run Clippy with warnings as errors
make check    # Run combined local checks
make examples # Build and run all examples
make docs     # Build the API reference
make preview  # Build and open the API reference in a browser
make package  # Verify packaging with cargo publish --dry-run
make clean    # Remove build artifacts
```

## Development

- Keep changes focused, follow nearby code, and preserve unrelated user changes.
- When changing atoms, keep shape, sign, curvature, numerical evaluation, and
  canonicalization consistent. Update public exports, tests, and documentation.
- Preserve column-major matrix ordering and Clarabel's `A x + s = b` convention.
- Enforce variable sign declarations as bounds and preserve explicit constraint
  dual mappings when adding auxiliary constraints.
- Start project-authored Rust files (`src/`, `tests/`, and `examples/`) with this
  header. New files use their creation year; preserve existing years and
  third-party notices when moving or updating code.

```rust
/*
Copyright <year> CVXRust Authors
SPDX-License-Identifier: Apache-2.0
*/
```

## Validation

- Start with focused checks, such as `cargo test <test_name>`. Cover changed
  behavior and relevant domain, shape, and failure cases.
- Use tolerances for floating-point assertions. Prefer mathematical behavior
  checks to source-text or fixed-prose matching.
- Use `make check` for formatting, lint, tests, examples, and documentation.
  Run `make package` separately when preparing a release.
- Choose checks appropriate to the change; prose-only instruction edits need no
  Rust build. Repeat successful checks only when further changes warrant it.
- Report the checks performed and any remaining failures or omissions.

## Documentation

- Write concise, precise English and standard mathematical notation. Avoid
  unnecessary jargon, promotional language, and defensive commentary.
- Follow Boyd and Vandenberghe's *Convex Optimization* notation where applicable.
- Explain problems and mathematical steps in execution order. Prefer vector and
  matrix notation when it improves clarity.
- Keep examples minimal. Explain non-obvious assumptions, domain restrictions,
  and reformulation steps.
- Update documentation sources and keep them consistent with the public API.
  Do not edit generated documentation.
- Use `make docs` to build the API reference or `make preview` to open it.
  Validate changed Rust documentation examples with `cargo test --doc` and run
  affected examples with `cargo run --example <name>`.
