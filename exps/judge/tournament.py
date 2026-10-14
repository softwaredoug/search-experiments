from dataclasses import dataclass

import networkx as nx


class NoOpScheduler:
    """Schedule the full population."""

    def __init__(self, all_nodes, n_per_round: int | None = None):
        self.all_nodes = set(all_nodes)

    def add_preference(self, winner: int, loser: int):
        pass

    def schedule(self) -> set[int]:
        return self.all_nodes


class TournamentScheduler:
    """Tournament adjacency graph for Blitzrank-style pairwise comparisons."""

    def __init__(self, all_nodes=(), n_per_round: int = 3):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(all_nodes)
        self.n_per_round = n_per_round

    def add_preference(self, winner: int, loser: int):
        self.graph.add_edge(winner, loser)

    def schedule(self) -> set[int]:
        """Schedule based on blitzrank rules, returning which pairs should be compared."""
        # Get raw stats for each SCC in dag, apply to each doc
        stats = {}
        dag = nx.condensation(self.graph)
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
                    "wins": wins,
                    "scc_wins": wins - internal,
                    "losses": losses,
                    "scc_losses": losses - internal,
                    "kappa": kappa,
                    "resolved": kappa == len(self.graph) - 1,
                }

        # Remove cycles from the graph to create a DAG
        members = nx.get_node_attributes(dag, "members")

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
        tournament = []
        for scc in unresolved[:self.n_per_round]:
            doc = min(
                members[scc],
                key=lambda d: (stats[d]["kappa"], d)
            )
            tournament.append(doc)

        return set(tournament)


@dataclass(frozen=True)
class SchedulerConfig:
    scheduler_class: type
    scheduler_type: str
    n_per_round: int | None
    max_rounds: int
    params: dict

    def create(self, all_nodes, *, default_n_per_round: int):
        n_per_round = (
            self.n_per_round
            if self.n_per_round is not None
            else default_n_per_round
        )
        return self.scheduler_class(all_nodes, n_per_round=n_per_round)

    @property
    def cache_config(self) -> dict:
        return {"type": self.scheduler_type, "params": self.params}


def scheduler_from_config(config: dict | None) -> SchedulerConfig:
    """Validate a scheduler block and return its reusable scheduler settings."""
    scheduler_type = "noop"
    params = {}
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError("strategy.params.scheduler must be a mapping.")
        scheduler_type = config.get("type")
        params = config.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("strategy.params.scheduler.params must be a mapping.")

    if scheduler_type == "noop":
        if params:
            raise ValueError("The noop scheduler does not accept params.")
        return SchedulerConfig(
            scheduler_class=NoOpScheduler,
            scheduler_type=scheduler_type,
            n_per_round=None,
            max_rounds=1,
            params=params.copy(),
        )

    if scheduler_type != "tournament":
        raise ValueError("strategy.params.scheduler.type must be 'noop' or 'tournament'.")

    unknown_params = set(params) - {"n_per_round", "max_rounds"}
    if unknown_params:
        raise ValueError(
            "Unknown tournament scheduler params: " + ", ".join(sorted(unknown_params))
        )
    n_per_round = params.get("n_per_round", 3)
    max_rounds = params.get("max_rounds", 5)
    for name, value in (("n_per_round", n_per_round), ("max_rounds", max_rounds)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"strategy.params.scheduler.params.{name} must be a positive integer."
            )
    if n_per_round < 2:
        raise ValueError("strategy.params.scheduler.params.n_per_round must be at least 2.")

    return SchedulerConfig(
        scheduler_class=TournamentScheduler,
        scheduler_type=scheduler_type,
        n_per_round=n_per_round,
        max_rounds=max_rounds,
        params=params.copy(),
    )
