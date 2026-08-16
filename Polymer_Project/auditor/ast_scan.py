"""
Deterministic AST scanner for look-ahead and leakage patterns.

Why an abstract syntax tree and not a regular expression: `df.shift(-1)` and
`df.shift(-1)` inside a string literal or a comment look identical to a regex.
The parser knows the difference, knows that `.rolling(center=True)` is a keyword
argument rather than a substring, and gives exact line numbers for free.

Why deterministic rules and not only a language model: these patterns are
frequent, unambiguous and cheap to match exactly. Rules cost nothing, run
offline, and give the same answer on every run. The language model is reserved
for the semantic cases that no rule can express, and its findings are kept in a
separate section of the report.

Every finding carries real line numbers taken from the parser, so a finding can
never point at a line that does not exist.
"""

from __future__ import annotations

import ast

from auditor.schema import RULES, Finding

# --- naming conventions used by the heuristic rules -------------------------

TARGET_NAMES = {"y", "y_train", "y_test", "target", "targets", "label", "labels"}
TARGET_PREFIXES = ("future_", "fwd_", "forward_", "next_", "y_")
TEST_MARKERS = ("test", "val", "valid", "holdout", "oos", "out_of_sample")
PREPROCESSOR_MARKERS = (
    "scaler", "standardscaler", "minmax", "robustscaler", "normalizer",
    "pca", "svd", "selector", "selectkbest", "encoder", "imputer", "vectorizer",
)
POSITION_MARKERS = ("pos", "position", "signal", "weight", "alpha", "exposure")
RETURN_MARKERS = ("ret", "return", "pnl", "pct_change", "diff", "fwd")
BACKFILL_METHODS = {"bfill", "backfill"}

#: Attributes that expose METADATA rather than values. Aligning on ``y_test.index``
#: is not the same as using ``y_test``, and treating it as leakage was a real
#: false positive found on a trapped benchmark script.
METADATA_ATTRS = {
    "index", "columns", "shape", "size", "name", "dtype", "dtypes",
    "empty", "ndim", "axes",
}


def _name_of(node: ast.AST) -> str:
    """Best-effort dotted name of an expression, lowercased."""
    if isinstance(node, ast.Name):
        return node.id.lower()
    if isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}".lower()
    if isinstance(node, ast.Subscript):
        base = _name_of(node.value)
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return f"{base}[{key.value}]".lower()
        return base
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    return ""


def _contains_marker(text: str, markers) -> bool:
    return any(marker in text for marker in markers)


#: Builtins whose argument's VALUES are not consumed numerically. A target
#: passed to these is being counted, printed or introspected, not leaked.
NON_VALUE_FUNCS = {
    "str", "repr", "len", "getattr", "hasattr", "isinstance", "type", "print",
    "enumerate", "zip", "sorted", "reversed", "list", "set", "dict", "tuple",
    "id", "format", "iter", "next", "join",
}


def _is_metadata_reference(name: str) -> bool:
    """True for names like ``y_test.index`` or ``y_train.shape[0]``.

    These read the shape or the labels of the target, not its values, so they
    are alignment code rather than leakage.
    """
    return any(part in METADATA_ATTRS for part in name.split("."))


def _negative_constant(node: ast.AST) -> bool:
    """True for a literal negative number, including unary minus."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value < 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        return isinstance(operand, ast.Constant) and isinstance(
            operand.value, (int, float)
        )
    return False


def _keyword(call: ast.Call, name: str):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


class LeakageVisitor(ast.NodeVisitor):
    """Walks the tree once and collects findings."""

    def __init__(self, source: str, filename: str = "") -> None:
        self.source = source
        self.filename = filename
        self.findings: list[Finding] = []
        self._lines = source.splitlines()
        #: names bound to a windowed object, e.g. ``roll = s.rolling(30)``.
        #: Without this, a z-score written through such a variable looks like a
        #: whole-sample normalisation and R10 fires on correct causal code.
        self._windowed_names: set[str] = set()
        #: > 0 while inside an assignment whose bare name is target vocabulary
        #: (``target = ...``, ``y_next = ...``). R1's own fix text says a
        #: negative shift "is only legitimate when building the TARGET"; this
        #: implements the exemption the rule was already prescribing. Found by
        #: seeding the benchmark: trap07 builds its label with shift(-1) and
        #: R1 flagged the one construction its documentation calls legitimate.
        #: The exemption is deliberately NARROW -- a bare target-named
        #: variable only. ``df['target'] = ...`` still fires: when unsure, a
        #: leak detector keeps firing. If the label later feeds a feature,
        #: that is R5's job, and R5 still fires on it.
        self._label_assignment_depth: int = 0

    # -- helpers ------------------------------------------------------------

    def _snippet(self, node: ast.AST) -> str:
        segment = ast.get_source_segment(self.source, node)
        if segment:
            return segment.strip().splitlines()[0][:160]
        line = getattr(node, "lineno", 1) - 1
        return self._lines[line].strip()[:160] if 0 <= line < len(self._lines) else ""

    def _add(self, rule_id: str, node: ast.AST, detail: str = "") -> None:
        rule = RULES[rule_id]
        explanation = rule.explanation if not detail else f"{rule.explanation} {detail}"
        self.findings.append(
            Finding(
                rule_id=rule.rule_id,
                title=rule.title,
                severity=rule.severity,
                line_start=getattr(node, "lineno", 1),
                line_end=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                snippet=self._snippet(node),
                explanation=explanation,
                suggested_fix=rule.suggested_fix,
                detector="ast",
                filename=self.filename,
            )
        )

    # -- visitors -----------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        attr = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        full = _name_of(node.func)

        # R1 -- negative shift (exempt inside a bare target-named assignment:
        # building the label is the one legitimate negative shift, per the
        # rule's own fix text; feeding it back in is R5's territory)
        if attr == "shift":
            arg = node.args[0] if node.args else _keyword(node, "periods")
            if arg is not None and _negative_constant(arg):
                if self._label_assignment_depth == 0:
                    self._add("R1", node)

        # R2 -- centred rolling window
        if attr in {"rolling", "rolling_mean"} and _is_true(_keyword(node, "center")):
            self._add("R2", node)

        # R8 -- backward fill or interpolation
        if attr in BACKFILL_METHODS:
            self._add("R8", node)
        elif attr == "fillna":
            method = _keyword(node, "method")
            if isinstance(method, ast.Constant) and str(method.value).lower() in BACKFILL_METHODS:
                self._add("R8", node)
        elif attr == "interpolate":
            self._add(
                "R8", node,
                "Interpolation is bidirectional unless limit_direction is set to "
                "'forward'.",
            )
        elif attr == "reindex":
            method = _keyword(node, "method")
            if isinstance(method, ast.Constant) and str(method.value).lower() in (
                BACKFILL_METHODS | {"nearest"}
            ):
                self._add(
                    "R8", node,
                    f"reindex(method={method.value!r}) pulls values backwards in time.",
                )

        # R3 / R9 -- fitting
        if attr in {"fit", "fit_transform"}:
            fitted_on = [_name_of(a) for a in node.args]
            target = _name_of(node.func.value) if isinstance(node.func, ast.Attribute) else ""

            if any(_contains_marker(a, TEST_MARKERS) for a in fitted_on):
                self._add(
                    "R9", node,
                    f"Fitted on {', '.join(a for a in fitted_on if a) or 'a test-named variable'}.",
                )
            elif _contains_marker(target, PREPROCESSOR_MARKERS) or _contains_marker(
                full, PREPROCESSOR_MARKERS
            ):
                if not any(
                    _contains_marker(a, ("train", "_tr", "insample", "in_sample"))
                    for a in fitted_on
                ):
                    self._add(
                        "R3", node,
                        "The argument is not visibly restricted to a training slice.",
                    )

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        targets = [_name_of(t) for t in node.targets]
        assigned = " ".join(t for t in targets if t)

        # Track handles to windowed objects before running any check on them.
        if self._is_windowed_expression(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._windowed_names.add(target.id.lower())

        # R5 -- feature built from the target
        looks_like_feature = not (
            assigned in TARGET_NAMES or assigned.startswith(TARGET_PREFIXES)
        )
        if looks_like_feature and assigned:
            hits = self._target_value_uses(node)
            if hits:
                self._add("R5", node, f"References {', '.join(sorted(hits))}.")

        # R10 -- normalisation by whole-sample statistics
        if self._is_whole_sample_normalisation(node.value):
            self._add("R10", node)

        # R4 -- possible same-bar execution
        if self._looks_like_unlagged_pnl(node.value):
            self._add("R4", node)

        is_label_assignment = assigned in TARGET_NAMES or assigned.startswith(
            TARGET_PREFIXES
        )
        if is_label_assignment:
            self._label_assignment_depth += 1
            self.generic_visit(node)
            self._label_assignment_depth -= 1
        else:
            self.generic_visit(node)

    # -- heuristics ---------------------------------------------------------

    WINDOW_METHODS = {"rolling", "expanding", "ewm", "groupby", "resample"}

    def _target_value_uses(self, assign: ast.Assign) -> set[str]:
        """Names of target-vocabulary references used as VALUES in the assignment.

        Being mentioned is not enough. ``" ".join(t for t in targets)`` iterates
        over a variable that happens to be called ``targets``; ``getattr(label,
        "date")`` introspects one called ``label``. Neither moves the target's
        VALUES into a feature, and both were real false positives found by
        running this scanner on its own source. A reference counts only when it
        feeds arithmetic, a comparison, a method chain, a subscript, or is the
        assigned value itself.
        """
        value = assign.value

        parents: dict[int, ast.AST] = {}
        for parent in ast.walk(assign):
            for child in ast.iter_child_nodes(parent):
                parents[id(child)] = parent

        def is_target_name(name: str) -> bool:
            return name in TARGET_NAMES or any(
                name.startswith(p) or f"[{p}" in name for p in TARGET_PREFIXES
            )

        def used_as_value(node: ast.AST) -> bool:
            cur = node
            while True:
                parent = parents.get(id(cur))
                if parent is None or isinstance(parent, (ast.Assign, ast.AugAssign)):
                    return True  # the reference IS the assigned value (an alias)
                if isinstance(parent, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp)):
                    return True
                if isinstance(parent, ast.Attribute) and parent.value is cur:
                    # y.index / y.shape read metadata, not values; y.rolling(...)
                    # and other method chains consume the values.
                    return parent.attr not in METADATA_ATTRS
                if isinstance(parent, ast.Subscript) and parent.value is cur:
                    return True  # y[...]
                if isinstance(parent, ast.comprehension):
                    return False  # the iterable or condition of a comprehension
                if isinstance(parent, ast.Call):
                    func = _name_of(parent.func).split(".")[-1]
                    if cur is parent.func:
                        return True
                    return func not in NON_VALUE_FUNCS
                cur = parent  # keyword, Lambda, IfExp, comprehension elt, ...

        hits: set[str] = set()
        for node in ast.walk(value):
            if not isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
                continue
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                continue  # binding occurrences (comprehension targets) are not uses
            name = _name_of(node)
            if not name or not is_target_name(name) or _is_metadata_reference(name):
                continue
            if used_as_value(node):
                hits.add(name)
        return hits

    def _is_windowed_expression(self, value: ast.AST) -> bool:
        """True if the expression builds or references a windowed object."""
        for n in ast.walk(value):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in self.WINDOW_METHODS
            ):
                return True
            if isinstance(n, ast.Name) and n.id.lower() in self._windowed_names:
                return True
        return False

    def _is_whole_sample_normalisation(self, value: ast.AST) -> bool:
        """(x - x.mean()) / x.std() with no rolling or expanding window involved.

        The numerator must contain a SUBTRACTION: a normalisation re-centres a
        level against its own statistic. Without that requirement, any ratio of
        two whole-sample statistics matches -- including ``mean() / std()``,
        which is a Sharpe ratio and a perfectly legitimate whole-sample
        computation. That exact false positive came out of running the scanner
        on this project's own source.
        """
        if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Div):
            return False
        numerator_has_sub = any(
            isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)
            for n in ast.walk(value.left)
        )
        if not numerator_has_sub:
            return False
        stat_calls = [
            n for n in ast.walk(value)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"mean", "std", "min", "max"}
        ]
        if len(stat_calls) < 2:
            return False
        return not self._is_windowed_expression(value)

    def _looks_like_unlagged_pnl(self, value: ast.AST) -> bool:
        """position * return over the same index, with no lag anywhere."""
        if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Mult):
            return False
        left, right = _name_of(value.left), _name_of(value.right)
        if not left or not right:
            return False
        pos_then_ret = _contains_marker(left, POSITION_MARKERS) and _contains_marker(
            right, RETURN_MARKERS
        )
        ret_then_pos = _contains_marker(right, POSITION_MARKERS) and _contains_marker(
            left, RETURN_MARKERS
        )
        if not (pos_then_ret or ret_then_pos):
            return False
        lagged = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"shift", "lag"}
            for n in ast.walk(value)
        )
        return not lagged


def audit_source(source: str, filename: str = "<string>") -> list[Finding]:
    """Parse and scan Python source, returning findings sorted by line."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return [
            Finding(
                rule_id="PARSE",
                title="File could not be parsed",
                severity="high",
                line_start=exc.lineno or 1,
                line_end=exc.lineno or 1,
                snippet=(exc.text or "").strip()[:160],
                explanation=f"Python could not parse this file: {exc.msg}.",
                suggested_fix="Fix the syntax error, then re-run the audit.",
                detector="ast",
                filename=filename,
            )
        ]

    visitor = LeakageVisitor(source, filename=filename)
    visitor.visit(tree)

    n_lines = len(source.splitlines())
    for finding in visitor.findings:
        assert 1 <= finding.line_start <= max(n_lines, 1), (
            f"{finding.rule_id} produced line {finding.line_start} in a file of "
            f"{n_lines} lines"
        )

    return sorted(visitor.findings, key=lambda f: (f.line_start, f.rule_id))


def audit_file(path: str) -> list[Finding]:
    with open(path, "r", encoding="utf-8") as handle:
        return audit_source(handle.read(), filename=path)


__all__ = ["audit_source", "audit_file", "LeakageVisitor"]
