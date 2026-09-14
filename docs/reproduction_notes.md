# Reproduction notes

This harness defaults to audited free generation. It stores exact prior assistant output, retains invalid/error items, and never gates revision by label correctness.

The official USC code parses failures with `continue`, compares forced one-token Yes/No logits locally, reconstructs some conversation content from correctness/labels, and writes mutable JSON. Those behaviors are deferred from this smoke slice. `legacy_forced_logits`, 300-item runs, plots, PACT, tuned lens, API backend, and wavering experiments are out of scope until smoke integrity passes.

All results must be labeled **adaptation** because model family, quantization, runtime/template, and subset differ from ACL 2025.
