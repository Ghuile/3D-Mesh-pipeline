# Contributing

Contributions should improve the clarity, reproducibility, or correctness of the research workflow. Identify the affected stage and distinguish documentation changes from experimental changes.

## Making a change

1. Create a branch from the revision you intend to update.
2. Keep the change focused and describe its purpose.
3. Update usage and reproducibility documentation.
4. Record notable changes under `Unreleased` in `CHANGELOG.md`.
5. Open a pull request with validation evidence.

Use descriptive names and concise comments explaining assumptions, units, indexing, and methodological choices. Document function inputs, outputs, and side effects where needed. Avoid guarantees of anatomical validity, convergence, or topology equivalence unless the implementation checks them.

## Validation

For documentation-only changes, run `git diff --check`, verify local links, and compare Python syntax trees after excluding docstrings. Preserve notebook code cells, outputs, execution counts, and existing metadata. Embedded Blender programs are executable strings and must also remain unchanged.

For code changes, report the script, dataset revision, environment, checkpoint, and command used. Include relevant before/after metrics and differences from the paper. Use Blender or Unreal Editor for scripts requiring their APIs.

There is no tracked unit-test suite. The `test_noise_robustness_*.py` scripts are experiments requiring data and checkpoints. Do not assume scripts implement `--help` or `--test`. Avoid importing scripts simply to check syntax: some execute processing at module scope.

## Data and artifacts

Keep raw meshes, generated arrays, checkpoints, local environments, and experiment outputs out of commits. Generation scripts and the existing population CSV are tracked source materials; preserve those exceptions in ignore rules.

Record dataset revisions, processing order, units, seeds, environment versions, and checkpoint provenance. Describe synthetic targets accurately.

## Pull requests

Include the problem, resulting behavior, affected stages, validation performed, and checks not performed. Explain effects on paper results, artifacts, dependencies, or configuration. Changes to numerical parameters, architecture, data ordering, or preprocessing must not be hidden in a comment cleanup.

## License and support

Preserve attribution and license notices. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Report reproducible problems through [GitHub issues](https://github.com/Ghuile/3D-Mesh-pipeline/issues), including the relevant revision and environment.
