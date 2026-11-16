"""observation.traced_metadata: traced construction derives static metadata from the argument form (spec 3.2, 12.1; rev 62).

Spec 3.2, rev 62: "When any observation input is traced, the flags follow the
constructor's argument form: only whether each observation argument is `None`
or supplied counts, never its values (traced or concrete). A flag is false only
in these cases; otherwise it is conservatively true:

- Flag form: with no `event` and no censoring flag, only the exact kind flag is
  true. With `event`, or with `right_censored` as the only censoring flag
  supplied, the left- and interval-censored flags are false. Supplying
  `left_censored` or `interval_censored` makes all four kind flags true.
- Bounds form: with `exact` omitted, the exact flag is false; the three
  censored kind flags are always true.
- Both forms: the left-truncation flag is false when `left_trunc` is omitted,
  and the right-truncation flag is false when `right_trunc` is omitted.

These rules hold for every row, including rows with issue bits."

The flags have no public accessor (as in `test_a3_static_metadata.py`), so each
test compares the tree structure of traced-built data with eager data whose
value-based flags (spec 3.2, rev 6) are exactly the expected set, and the
compile-count tests observe the same through `jit` retracing. Before this
feature, traced flags are all true; those tests in `test_a3_static_metadata.py`
carry `until("observation.traced_metadata")`.
"""

import jax
import numpy as np
import pytest
from helpers import INF, NAN, b, f, treedef

import survjax

FEATURE = pytest.mark.feature("observation.traced_metadata")


# ------------------------------------------------- eager references (value-based flags)


def _eager_exact(**trunc):
    """Exact rows only; ``trunc`` adds truncation that sets the matching flag."""
    return survjax.SurvivalData(f([1.0, 2.0, 3.0]), **trunc)


def _eager_exact_right(**trunc):
    return survjax.SurvivalData(f([1.0, 2.0, 3.0]), right_censored=b([False, True, False]), **trunc)


def _eager_all_kinds(**trunc):
    """Exact 1, right at 2, left at 3, interval (1, 4]."""
    return survjax.SurvivalData(
        f([1.0, 2.0, 3.0, 1.0]),
        right_censored=b([False, True, False, False]),
        left_censored=b([False, False, True, False]),
        interval_censored=b([False, False, False, True]),
        time2=f([NAN, NAN, NAN, 4.0]),
        **trunc,
    )


def _eager_censored_kinds(**trunc):
    """Bounds form without exact rows: left `(0, 2]`, right `(1, inf]`, interval `(1, 3]`."""
    return survjax.SurvivalData.from_bounds(f([0.0, 1.0, 1.0]), f([2.0, INF, 3.0]), **trunc)


# Truncation that the eager references need to set each flag from values.
LT3 = {"left_trunc": f([0.5, 0.0, 0.0])}
RT3 = {"right_trunc": f([10.0, INF, INF])}
LT4 = {"left_trunc": f([0.5, 0.0, 0.0, 0.0])}
RT4 = {"right_trunc": f([10.0, INF, INF, INF])}


# ------------------------------------------------- traced construction


def _traced_flag(**kw):
    """Flag form under `jit`, every given argument a traced `jit` argument."""
    names = list(kw)

    def build(*values):
        args = dict(zip(names, values, strict=True))
        time = args.pop("time")
        event = args.pop("event", None)
        if event is None:
            return survjax.SurvivalData(time, **args)
        return survjax.SurvivalData(time, event, **args)

    return jax.jit(build)(*kw.values())


def _traced_bounds(**kw):
    """Bounds form under `jit`, every given argument a traced `jit` argument."""
    names = list(kw)

    def build(*values):
        args = dict(zip(names, values, strict=True))
        return survjax.SurvivalData.from_bounds(args.pop("lower"), args.pop("upper"), **args)

    return jax.jit(build)(*kw.values())


T3 = f([1.0, 2.0, 3.0])
ALL_FALSE3 = b([False, False, False])
ALL_TRUE3 = b([True, True, True])


# ------------------------------------------------- flag form: kind flags


@FEATURE
def test_traced_time_alone_has_only_the_exact_flag():
    """spec 3.2, rev 62: "with no `event` and no censoring flag, only the exact kind flag is true"; no truncation flags."""
    assert treedef(_traced_flag(time=T3)) == treedef(_eager_exact())


@FEATURE
def test_traced_time_alone_is_not_all_kinds():
    """spec 3.2, rev 62: the exact-only form no longer gives conservative all-true flags."""
    assert treedef(_traced_flag(time=T3)) != treedef(_eager_all_kinds(**LT4, **RT4))


@FEATURE
def test_traced_event_gives_exact_and_right_flags():
    """spec 3.2, rev 62: "With `event` ... the left- and interval-censored flags are false"; exact and right stay true."""
    data = _traced_flag(time=T3, event=b([True, False, True]))
    assert treedef(data) == treedef(_eager_exact_right())


@FEATURE
def test_traced_event_flags_ignore_values():
    """spec 3.2, rev 62: "only whether each observation argument is `None` or supplied counts": all events observed still gives exact and right."""
    data = _traced_flag(time=T3, event=ALL_TRUE3)
    assert treedef(data) == treedef(_eager_exact_right())


@FEATURE
def test_traced_right_censored_alone_gives_exact_and_right_flags():
    """spec 3.2, rev 62: `right_censored` "as the only censoring flag supplied" keeps left and interval false."""
    data = _traced_flag(time=T3, right_censored=b([False, True, False]))
    assert treedef(data) == treedef(_eager_exact_right())


@FEATURE
def test_traced_all_false_right_censored_still_sets_the_right_flag():
    """spec 3.2, rev 62: the right flag follows the supplied argument, not its values (all `False` here)."""
    data = _traced_flag(time=T3, right_censored=ALL_FALSE3)
    assert treedef(data) == treedef(_eager_exact_right())


@FEATURE
def test_traced_left_censored_sets_all_four_kind_flags():
    """spec 3.2, rev 62: "Supplying `left_censored` ... makes all four kind flags true" (values all `False`)."""
    data = _traced_flag(time=T3, left_censored=ALL_FALSE3)
    assert treedef(data) == treedef(_eager_all_kinds())


@FEATURE
def test_traced_interval_censored_sets_all_four_kind_flags():
    """spec 3.2, rev 62: "Supplying ... `interval_censored` makes all four kind flags true" (values all `False`)."""
    data = _traced_flag(time=T3, interval_censored=ALL_FALSE3, time2=f([NAN, NAN, NAN]))
    assert treedef(data) == treedef(_eager_all_kinds())


@FEATURE
def test_traced_right_and_left_censored_set_all_four_kind_flags():
    """spec 3.2, rev 62: `right_censored` is not "the only censoring flag supplied" once `left_censored` is too."""
    data = _traced_flag(time=T3, right_censored=b([False, True, False]), left_censored=ALL_FALSE3)
    assert treedef(data) == treedef(_eager_all_kinds())


# ------------------------------------------------- flag form: truncation flags


@FEATURE
def test_traced_left_trunc_sets_only_the_left_truncation_flag():
    """spec 3.2, rev 62: `left_trunc` supplied (all zero) sets left truncation; `right_trunc` omitted keeps right truncation false."""
    data = _traced_flag(time=T3, left_trunc=f([0.0, 0.0, 0.0]))
    assert treedef(data) == treedef(_eager_exact(**LT3))


@FEATURE
def test_traced_right_trunc_sets_only_the_right_truncation_flag():
    """spec 3.2, rev 62: `right_trunc` supplied (all `+inf`) sets right truncation; `left_trunc` omitted keeps left truncation false."""
    data = _traced_flag(time=T3, right_trunc=f([INF, INF, INF]))
    assert treedef(data) == treedef(_eager_exact(**RT3))


@FEATURE
def test_traced_both_truncations_set_both_flags():
    """spec 3.2, rev 62: both bounds supplied set both truncation flags, with the event form's kind flags."""
    data = _traced_flag(
        time=T3, event=ALL_TRUE3, left_trunc=f([0.0, 0.0, 0.0]), right_trunc=f([INF, INF, INF])
    )
    assert treedef(data) == treedef(_eager_exact_right(**LT3, **RT3))


@FEATURE
def test_traced_left_trunc_alone_gives_a_different_structure_from_right_trunc_alone():
    """spec 3.2, rev 62: the two truncation flags are separate under tracing too."""
    left = _traced_flag(time=T3, left_trunc=f([0.0, 0.0, 0.0]))
    right = _traced_flag(time=T3, right_trunc=f([INF, INF, INF]))
    assert treedef(left) != treedef(right)


@FEATURE
def test_every_argument_supplied_gives_all_flags():
    """spec 3.2, rev 62: with every observation argument supplied, every flag is true (the conservative case)."""
    data = _traced_flag(
        time=f([1.0, 2.0, 3.0, 1.0]),
        right_censored=b([False, True, False, False]),
        left_censored=b([False, False, True, False]),
        interval_censored=b([False, False, False, True]),
        time2=f([NAN, NAN, NAN, 4.0]),
        left_trunc=f([0.5, 0.5, 0.0, 0.0]),
        right_trunc=f([10.0, INF, 10.0, INF]),
    )
    assert treedef(data) == treedef(_eager_all_kinds(**LT4, **RT4))


# ------------------------------------------------- concrete arguments next to a traced one


@FEATURE
def test_concrete_censoring_flag_counts_by_form_when_time_is_traced():
    """spec 3.2, rev 62: "never its values (traced or concrete)": a concrete all-`False` `right_censored` beside traced `time` still sets right."""
    flags = ALL_FALSE3
    data = jax.jit(lambda t: survjax.SurvivalData(t, right_censored=flags))(T3)
    assert treedef(data) == treedef(_eager_exact_right())


@FEATURE
def test_concrete_left_trunc_counts_by_form_when_time_is_traced():
    """spec 3.2, rev 62: a concrete all-zero `left_trunc` beside traced `time` sets the left-truncation flag."""
    lt = f([0.0, 0.0, 0.0])
    data = jax.jit(lambda t: survjax.SurvivalData(t, left_trunc=lt))(T3)
    assert treedef(data) == treedef(_eager_exact(**LT3))


@FEATURE
def test_traced_truncation_beside_concrete_time_counts_by_form():
    """spec 3.2, rev 62: any traced observation input switches to form rules: traced `left_trunc`, concrete `time`, gives exact and left truncation."""
    data = jax.jit(lambda lt: survjax.SurvivalData(T3, left_trunc=lt))(f([0.0, 0.0, 0.0]))
    assert treedef(data) == treedef(_eager_exact(**LT3))


# ------------------------------------------------- rows with issue bits


@FEATURE
def test_traced_rows_with_issue_bits_keep_the_form_flags():
    """spec 3.2, rev 62: "These rules hold for every row, including rows with issue bits": a negative and a NaN time leave exact-only flags."""
    data = _traced_flag(time=f([1.0, -1.0, NAN]))
    assert not bool(np.asarray(data.issues.ok))
    assert treedef(data) == treedef(_eager_exact())


@FEATURE
def test_traced_invalid_event_codes_keep_the_event_form_flags():
    """spec 3.2, rev 62: an event form with an invalid code (issue bit) still gives exact and right only."""
    data = _traced_flag(time=T3, event=np.array([1, 2, 0], dtype=np.int32))
    assert not bool(np.asarray(data.issues.ok))
    assert treedef(data) == treedef(_eager_exact_right())


# ------------------------------------------------- bounds form


@FEATURE
def test_traced_bounds_without_exact_have_no_exact_flag():
    """spec 3.2, rev 62: "with `exact` omitted, the exact flag is false; the three censored kind flags are always true"."""
    data = _traced_bounds(lower=f([1.0, 2.0, 3.0]), upper=f([4.0, 5.0, 6.0]))
    assert treedef(data) == treedef(_eager_censored_kinds())


@FEATURE
def test_traced_bounds_without_exact_keep_all_censored_flags_for_one_kind_of_row():
    """spec 3.2, rev 62: the censored flags do not depend on values: right-censored rows only still give left, right and interval."""
    data = _traced_bounds(lower=f([1.0, 2.0, 3.0]), upper=f([INF, INF, INF]))
    assert treedef(data) == treedef(_eager_censored_kinds())


@FEATURE
def test_traced_bounds_with_exact_set_all_four_kind_flags():
    """spec 3.2, rev 62: `exact` supplied (all `False`) makes the exact flag true beside the three censored flags."""
    data = _traced_bounds(lower=f([1.0, 2.0, 3.0]), upper=f([4.0, 5.0, 6.0]), exact=ALL_FALSE3)
    assert treedef(data) == treedef(_eager_all_kinds())


@FEATURE
def test_traced_bounds_truncation_flags_follow_the_supplied_bounds():
    """spec 3.2, rev 62: in the bounds form too, only a supplied `left_trunc` sets left truncation."""
    data = _traced_bounds(lower=f([1.0, 2.0, 3.0]), upper=f([4.0, 5.0, 6.0]), left_trunc=f([0.0, 0.0, 0.0]))
    assert treedef(data) == treedef(_eager_censored_kinds(**LT3))


@FEATURE
def test_traced_bounds_with_both_truncations():
    """spec 3.2, rev 62: both truncation bounds supplied set both flags in the bounds form."""
    data = _traced_bounds(
        lower=f([1.0, 2.0, 3.0]),
        upper=f([4.0, 5.0, 6.0]),
        left_trunc=f([0.0, 0.0, 0.0]),
        right_trunc=f([INF, INF, INF]),
    )
    assert treedef(data) == treedef(_eager_censored_kinds(**LT3, **RT3))


# ------------------------------------------------- slices and compile counts


@FEATURE
def test_slice_of_traced_data_inherits_the_form_flags():
    """spec 3.2, rev 62: "Slices inherit them unchanged"; a slice of traced event-form data keeps exact and right."""
    data = _traced_flag(time=T3, event=b([True, False, True]))
    assert treedef(data[0:2]) == treedef(_eager_exact_right())


def _counting_jit():
    traces = []

    @jax.jit
    def total(data):
        traces.append(None)
        return data.upper.sum()

    return total, traces


@FEATURE
def test_traced_and_eager_data_with_equal_flags_share_one_compilation():
    """spec 3.2, rev 62: traced event-form data and eager exact/right data have equal flags and compile once."""
    total, traces = _counting_jit()
    total(_eager_exact_right())
    total(_traced_flag(time=T3, event=b([True, False, True])))
    assert len(traces) == 1


@FEATURE
def test_traced_forms_with_different_flags_compile_separately():
    """spec 3.2, rev 62: "datasets with different flags ... compile separately": time-only and event-form traced data."""
    total, traces = _counting_jit()
    total(_traced_flag(time=T3))
    total(_traced_flag(time=T3, event=b([True, False, True])))
    assert len(traces) == 2


@FEATURE
def test_repeated_traced_form_reuses_compilation_for_new_values():
    """spec 3.2/7.2, rev 62: new values in the same argument form give the same flags, so no retrace."""
    total, traces = _counting_jit()
    total(_traced_flag(time=T3, event=b([True, False, True])))
    total(_traced_flag(time=f([4.0, 5.0, 6.0]), event=ALL_TRUE3))
    assert len(traces) == 1
