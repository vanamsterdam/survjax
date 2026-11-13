# DOCS.TRUNC.FLAGS handoff

Independent review: round 1/3

Read docs/likelihood.md's censored and truncated example first, then this
step's plan. The flags represent the same four sets and inclusion window as
the preceding bounds example; time2 is read only for interval rows.

The executable example checks its likelihood against the existing independent
exponential algebra (spec sections 3 and 6). scripts/build_docs.sh succeeded
in both dtype modes. scripts/check.sh --fast passed all gates; the existing
spec test for PH grid values emitted two NumPy runtime warnings. No standalone
tests or public API changes. No known uncertainty or remaining sub-steps.
