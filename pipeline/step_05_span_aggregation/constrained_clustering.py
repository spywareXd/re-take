"""
Constrained clustering for Step 5. User labels are constraints, not scores:
    "retake"    -> must-link   (processed first, always tried before any model edge)
    "different" -> cannot-link (a union is skipped if it would put a cannot-link pair into one component)
Edges (model-confident pairs) are then processed in descending score. The result is a set of disjoint components, so a sentence is in at most
one group; components with fewer than `min_size` sentences are dropped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Hashable, Iterable

logger = logging.getLogger(__name__)


@dataclass
class Edge:
    a: Hashable
    b: Hashable
    score: float = 0.0
    kind: str = "model"                 # "model" | "user"
    tag: object = None                  # caller payload (e.g. the pair), returned untouched with the accepted edges


@dataclass
class ClusterResult:
    components: list[list] = field(default_factory=list)       # members of every component with >= min_size nodes
    accepted: list[Edge] = field(default_factory=list)         # edges that were merged (or already inside one component)
    rejected: list[Edge] = field(default_factory=list)         # edges skipped because a cannot-link forbids the union


def cluster_with_constraints(
    edges: Iterable[Edge],
    must_link: Iterable[Edge] = (),
    cannot_link: Iterable[tuple] = (),
    min_size: int = 2,
) -> ClusterResult:
    parent: dict = {}
    members: dict = {}
    cannot: dict = {}
    for a, b in cannot_link:
        cannot.setdefault(a, set()).add(b)
        cannot.setdefault(b, set()).add(a)

    def find(x):
        if x not in parent:
            parent[x], members[x] = x, {x}
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def forbidden(ra, rb) -> bool:
        small, large = (members[ra], members[rb]) if len(members[ra]) <= len(members[rb]) else (members[rb], members[ra])
        return any(x in cannot and not cannot[x].isdisjoint(large) for x in small)

    res = ClusterResult()

    def process(e: Edge) -> None:
        if e.a == e.b:
            return
        ra, rb = find(e.a), find(e.b)
        if ra == rb:
            res.accepted.append(e)
            return
        if forbidden(ra, rb):                                    # checked BEFORE merging the two components
            res.rejected.append(e)
            logger.info("[Cluster] union %s - %s skipped: a cannot-link pair would end up in one component", e.a, e.b)
            return
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        members[ra] |= members.pop(rb)
        res.accepted.append(e)

    for e in sorted(must_link, key=lambda e: (-e.score, str(e.a), str(e.b))):
        process(e)
    for e in sorted(edges, key=lambda e: (-e.score, str(e.a), str(e.b))):
        process(e)

    res.components = [sorted(m, key=str) for r, m in members.items() if r == parent[r] and len(m) >= min_size]
    return res


def pair_key(unit_a: dict, unit_b: dict) -> str:
    """'<earlier>|<later>' by start time (ties by id): the pair id used in the review documents and in the labels."""
    a, b = sorted((unit_a, unit_b), key=lambda u: (float(u.get("start", 0.0)), str(u["unit_id"])))
    return f"{a['unit_id']}|{b['unit_id']}"
