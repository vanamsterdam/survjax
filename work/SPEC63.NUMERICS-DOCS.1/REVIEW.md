# SPEC63.NUMERICS-DOCS.1 — Document tail value and derivative accuracy

Independent review: round 1/3

## Read first

1. PLAN.md and preceding [SPEC63.LOG1MEXP.1](../SPEC63.LOG1MEXP.1/REVIEW.md).
2. docs/numerical-behaviour.md: the general zero-mass explanation and
   "Precision in both tails" section, including the two late-time cells.
3. Spec 4.1–4.2, 7.1–7.3 and 12.1, revision 63.

## Change and invariants

The page now distinguishes lower-tail relative log-CDF value accuracy from
upper-tail absolute accuracy. It explains probability-near-one rounding,
continued relative parameter-derivative accuracy on guaranteed smooth cases,
log_survival as the upper-tail representation, and structural uninformative
diagnostics. The general zero-mass paragraph no longer contradicts that
rounding rule. The existing section anchor and lower-tail example remain.

The late Weibull example changes from t=60 to t=70 months, making H=49 for
k=2 and b=10. It shows zero returned log F, positive survival and log S near
-49. A new cell differentiates log F in the scale, demonstrating a nonzero
derivative even though the returned value is zero. Expected values in prose
come from hand algebra: S=exp(-49), and d/db log F=-(2H/b)S/(1-S).
No API, library code, dependency, test or protected input changes.

This completes all Docs to-do items in SPEC63.LOG1MEXP.1's linked plan. It does
not activate any additional feature or promise new boundary derivatives.

## Verification

Ran `scripts/build_docs.sh` directly, without redirects or environment
assignments. Both dtype builds succeeded with Sphinx -E -n -W --keep-going,
executing all 26 pages. The changed notebook executed in 5.56s (float32) and
3.37s (float64). The build-identity check ended with `state: success` and
`Verified both dtypes`; source and Git state stayed unchanged during the build.
The build is correctly labelled dirty at starting commit 3701148, containing
the planned page edit and this step's PLAN.md. Generated HTML is ignored.

Inspected generated HTML and executed outputs using Python's HTMLParser:
the precision-in-both-tails anchor occurs once per mode, the updated accuracy
and derivative explanations are present, and the final cells contain:

| Output | Float32 | Float64 |
| --- | --- | --- |
| log F(70) | 0 | 0 |
| S(70) | 5.2429655e-22 | 5.24288566e-22 |
| log S(70) | -48.999985 | -49 |
| scale derivative | -5.1381043e-21 | -5.13802795e-21 |

The outputs agree with the stated hand formulas at ordinary dtype accuracy.
The lower-tail cell and all other existing cells also executed successfully.
No new unit tests were added for this prose-only step. The preceding code
step records the mpmath, derivative/Hessian and full acceptance evidence.

Visual browser QA remains unverified. Browser access reported an unavailable
host; the native Safari fallback was unusually slow and then reported that the
user had changed the app before further input. Inspection stopped at that
point. No rendered-page screenshot or MathJax visual result is claimed.

`scripts/check.sh --fast` passes with the entire baseline and all five committed
features enabled:

| Stage | Float64 | Float32 |
| --- | --- | --- |
| Implementer tests | 1411 passed, 8 skipped | 1410 passed, 9 skipped |
| Source doctests | 4 passed | 4 passed |
| Spec tests | 4827 passed, 46 skipped | 4761 passed, 102 skipped |

Ruff: 308 files formatted, lint passed. Mypy: no issues in 25 source files.
Each spec run has the same two existing NumPy PH reference warnings. The
gate ends with `check: all gates passed`. Full non-fast gate was not repeated
for this docs-only step; the preceding code step records its passing result.
`git diff --check` passes.
No R fixture regeneration, new tolerance, or spec/test edit was performed.

## Handoff

Starting and previous-step final commit:
3701148319c988f29b3caa76425f68b5f3e9c004. Previous implementation reviewed:
69ffc7c. Original branch/worktree base:
3852184bb0e6f0a0a9b746ad8b92a95f675b23d0. No new deviations.
No further substeps in this documentation topic; maintainer integration remains.
Independent review and final diff display pending.

ROADMAP status: Revision-63 numerical-behaviour documentation updated and
executed successfully in both dtypes; fast gate passes. Independent review and
maintainer integration pending; no documentation substeps remain for this topic.
