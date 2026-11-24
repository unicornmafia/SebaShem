"""Put source line markers into model output, using the decoder's attention.

For a marker before source token ``k``, the output is cut at the position ``p``
that leaves the fewest output tokens on the wrong side: tokens before ``p`` that
attend to source tokens at or after ``k``, plus tokens from ``p`` on that attend
before ``k``. This holds up under reordering (English vs. Egyptian word order),
where "first output token aligned past k" would not. Cuts are only made where
``allowed`` says (word boundaries) and never go backwards for later markers.
"""

from __future__ import annotations


def best_cut(alignment: list[int], k: int, allowed: list[int], start: int = 0) -> int:
    n = len(alignment)
    # cost(p) = #{i < p : a_i >= k} + #{i >= p : a_i < k}
    after_before = [0] * (n + 1)
    for i, a in enumerate(alignment):
        after_before[i + 1] = after_before[i] + (a >= k)
    before_total = sum(a < k for a in alignment)
    best, best_cost = None, None
    for p in allowed:
        if p < start:
            continue
        before_after = before_total - (p - after_before[p])  # tokens from p on that attend before k
        cost = after_before[p] + before_after
        if best_cost is None or cost < best_cost:
            best, best_cost = p, cost
    return n if best is None else best


def place_markers(alignment: list[int], source_positions: list[int], allowed: list[int]) -> list[int]:
    """Output cut position for each marker, in order."""
    cuts, start = [], 0
    for k in source_positions:
        p = best_cut(alignment, k, allowed, start)
        cuts.append(p)
        start = p
    return cuts


def best_source_cut(alignment: list[int], q: int, allowed: list[int]) -> int | None:
    """The inverse of :func:`best_cut`: the source cut matching output position ``q``.

    Chooses the source position ``p`` (from ``allowed``) that leaves the fewest
    output tokens on the wrong side: tokens before ``q`` attending at or after
    ``p``, plus tokens from ``q`` on attending before ``p``.
    """
    before, after = alignment[:q], alignment[q:]
    best, best_cost = None, None
    for p in allowed:
        cost = sum(a >= p for a in before) + sum(a < p for a in after)
        if best_cost is None or cost < best_cost:
            best, best_cost = p, cost
    return best
