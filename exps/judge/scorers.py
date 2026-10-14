from dataclasses import dataclass

import networkx as nx


_OUTCOMES = {"lhs", "rhs", "neither"}


class CopelandAllScorer:
    """Schedule all candidates and score every accumulated pairwise decision."""

    def __init__(self, all_nodes, n_per_round: int | None = None):
        del n_per_round
        self.all_nodes = set(all_nodes)
        self.points = dict.fromkeys(self.all_nodes, 0.0)

    def add_decision(self, lhs: int, rhs: int, outcome: str):
        if outcome not in _OUTCOMES:
            raise ValueError(f"Invalid pairwise outcome: {outcome!r}.")
        if outcome == "lhs":
            self.points[lhs] += 1.0
        elif outcome == "rhs":
            self.points[rhs] += 1.0
        else:
            self.points[lhs] += 0.5
            self.points[rhs] += 0.5

    def schedule(self) -> set[int]:
        return set(self.all_nodes)

    def score(self) -> tuple[list[int], list[float]]:
        ranked = sorted(self.points.items(), key=lambda item: (-item[1], item[0]))
        return [node for node, _ in ranked], [score for _, score in ranked]


class TournamentScorer:
    """Schedule unresolved SCC representatives and rank candidates by SCC tier."""

    def __init__(self, all_nodes=(), n_per_round: int = 3):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(all_nodes)
        self.n_per_round = n_per_round

    def add_decision(self, lhs: int, rhs: int, outcome: str):
        if outcome not in _OUTCOMES:
            raise ValueError(f"Invalid pairwise outcome: {outcome!r}.")
        if outcome == "lhs":
            self.graph.add_edge(lhs, rhs)
        elif outcome == "rhs":
            self.graph.add_edge(rhs, lhs)

    def _scc_stats(self):
        stats = {}
        dag = nx.condensation(self.graph)

        # SCC is just 'supernode' - a cycle of nodes that
        # acts as one node
        for scc in dag:
            members = dag.nodes[scc]["members"]
            internal = len(members) - 1

            losses = internal + sum(
                len(dag.nodes[a]["members"])
                for a in nx.ancestors(dag, scc)
            )

            wins = internal + sum(
                len(dag.nodes[d]["members"])
                for d in nx.descendants(dag, scc)
            )

            kappa = losses + wins - internal

            for doc_id in members:
                stats[doc_id] = {
                    "scc_wins": wins - internal,
                    "scc_losses": losses - internal,
                    "kappa": kappa,
                    "resolved": kappa == len(self.graph) - 1,
                }

        members = nx.get_node_attributes(dag, "members")
        return dag, members, stats

    def schedule(self) -> set[int]:
        """Return one representative from each highest-priority unresolved SCC."""
        dag, members, stats = self._scc_stats()

        # Any that do not have a direct or indirect relationship
        # to the rest of the graph
        unresolved = [
            scc for scc in dag
            if any(not stats[d]["resolved"] for d in members[scc])
        ]

        def scc_sort_key(scc):
            representative_id = min(members[scc])
            return (
                stats[representative_id]["scc_losses"],
                stats[representative_id]["scc_wins"],
                representative_id,
            )

        unresolved.sort(key=scc_sort_key)
        candidates = []
        for scc in unresolved[:self.n_per_round]:
            candidates.append(min(members[scc]))
        return set(candidates)

    def score(self) -> tuple[list[int], list[float]]:
        """Return SCC-tier scores, tying documents with identical reach metrics."""
        dag, members, stats = self._scc_stats()
        scc_metrics = {
            scc: (
                stats[min(members[scc])]["scc_losses"],
                stats[min(members[scc])]["scc_wins"],
            )
            for scc in dag
        }
        ordered_tiers = sorted(set(scc_metrics.values()))
        tier_scores = {
            metrics: float(len(ordered_tiers) - rank)
            for rank, metrics in enumerate(ordered_tiers)
        }
        scores = {
            doc_id: tier_scores[scc_metrics[scc]]
            for scc in dag
            for doc_id in members[scc]
        }
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [node for node, _ in ranked], [score for _, score in ranked]


@dataclass(frozen=True)
class ScorerConfig:
    scorer_class: type
    scorer_type: str
    n_per_round: int | None
    max_rounds: int
    params: dict

    def create(self, all_nodes, *, default_n_per_round: int):
        n_per_round = (
            self.n_per_round
            if self.n_per_round is not None
            else default_n_per_round
        )
        return self.scorer_class(all_nodes, n_per_round=n_per_round)

    @property
    def cache_config(self) -> dict:
        return {"type": self.scorer_type, "params": self.params}


def scorer_from_config(config: dict | None) -> ScorerConfig:
    """Validate a scorer block and return its reusable scorer settings."""
    scorer_type = "copeland_all"
    params = {}
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError("strategy.params.scorer must be a mapping.")
        scorer_type = config.get("type")
        params = config.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("strategy.params.scorer.params must be a mapping.")

    if scorer_type == "copeland_all":
        if params:
            raise ValueError("The copeland_all scorer does not accept params.")
        return ScorerConfig(
            scorer_class=CopelandAllScorer,
            scorer_type=scorer_type,
            n_per_round=None,
            max_rounds=1,
            params=params.copy(),
        )

    if scorer_type != "tournament":
        raise ValueError(
            "strategy.params.scorer.type must be 'copeland_all' or 'tournament'."
        )

    unknown_params = set(params) - {"n_per_round", "max_rounds"}
    if unknown_params:
        raise ValueError(
            "Unknown tournament scorer params: " + ", ".join(sorted(unknown_params))
        )
    n_per_round = params.get("n_per_round", 3)
    max_rounds = params.get("max_rounds", 5)
    for name, value in (("n_per_round", n_per_round), ("max_rounds", max_rounds)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"strategy.params.scorer.params.{name} must be a positive integer."
            )
    if n_per_round < 2:
        raise ValueError("strategy.params.scorer.params.n_per_round must be at least 2.")

    return ScorerConfig(
        scorer_class=TournamentScorer,
        scorer_type=scorer_type,
        n_per_round=n_per_round,
        max_rounds=max_rounds,
        params=params.copy(),
    )
