# PH.PERF.2 — Share PH survival evaluation for exact/right likelihoods

Step starting commit: 26f6b179c8740a28af03a49e9d9c9cb3068c9c87
Feature: issue [#8](https://github.com/vanamsterdam/survjax/issues/8), optimization.
Previous step: [PH.PERF.1](../PH.PERF.1/PLAN.md), ending at the starting commit.
Branch/worktree base: 54e3fa9f280f63bcd9c08865fc99e128cd69c370.
Dependencies: PH.PERF.1 measurements, existing metadata pruning and PH rules.
Completed feature selection remains scripts/spec-features.txt unchanged.

## Changes

Change src/survjax/likelihood.py only in the evaluator: when concrete type is
ProportionalHazards and static metadata permits both exact/right kinds but
excludes left/interval censoring and all truncation, evaluate PH log survival
once at each valid row's lower time. Add the exact-row PH log hazard evaluated
at separately protected density times; select zero for inactive hazards before
addition. Bad rows use safe substituted times and retain final NaNs. Preserve
parameter preparation, common dtype, validation, diagnostics and generic paths.
Restrict to the concrete PH type so subclass overrides remain honored.
No PH-equivalent delegation, primitive changes, public API, module, dependency
or protected-path edits. Spec: 2, 3.2–3.3, 4.4, 7.1–7.3 and 12.1–12.3.

Create tests/unit/test_ph_likelihood_shared_survival.py. Cover independent
interior values/Jacobians/Hessians for built-ins, zero-time uninformative rows,
generic-baseline inactive density safety, boundary and extreme values including
unrepresentable closures, invalid raw/current data and parameter isolation,
subclass fallback, metadata eligibility and compilation reuse. Existing PH
isolation/tail suites and wrapper censoring/truncation derivatives supplement
the focused file. Tests never assert a speed threshold or alter tolerances.

Extend scripts/benchmark_ph.py with an inherited registered PH control that
uses the unchanged generic evaluator (concrete-type guard excludes subclasses).
Interleave optimized PH and that control with the existing reference paths in
the same run. The inherited control retains the exact previous PH arithmetic;
it is a measurement tool, not new library API. Run in both dtypes, record
compile and warmed timings with independent oracle checks, and retain an
optimization only if it is worthwhile. Save raw outputs here.
Address PH.PERF.1 review nit N-1 with a platform.processor() fallback when
CPU-brand lookup is denied, so restricted environments can run smoke checks.

Target about 300 changed implementation lines across source, tests and script.
Run touched tests completely in both dtypes, refresh the slow list if a new
test exceeds one second, then scripts/check.sh --fast. The resulting 541-line
generated classification refresh is isolated in a mechanical commit, following
the B3.5 precedent; handwritten implementation is about 313 changed lines.
Write REVIEW.md, commit,
run independent-review (up to three rounds), display the step diff and stop.
If candidate measurements do not justify retaining the change, record that
decision instead of claiming an improvement. No following step without the
maintainer's instruction.

## Docs to-do and remaining scope

No public docs needed: behavior and API stay the same. General PH interval
specialization is outside this step; report any remaining measured gap and
whether issue #8's acceptance is satisfied by this focused optimization.
