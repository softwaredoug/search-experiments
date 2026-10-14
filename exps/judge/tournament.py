import networkx as nx


class NoOpScheduler:
    """Schedule the full population."""
    def __init__(self, all_nodes: set[int]):
        self.all_nodes = all_nodes

    def add_preference(self, winner: int, loser: int):
        pass

    def schedule(self, num_nodes: int | None = None) -> set[int]:
        return self.all_nodes


class TournamentScheduler:
    """Tournament adjacency graph for Blitzrank-style pairwise comparisons."""
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_preference(self, winner: int, loser: int):
        self.graph.add_edge(winner, loser)

    def schedule(self, num_nodes: int) -> set[int]:
        """Schedule based on blitzrank rules, returning which pairs should be compared."""
        # Remove cycles from the graph to create a DAG
        dag = nx.condensation(self.graph)

        # Track the size of any condensed nodes that
        # sit in a cycle
        sizes = {
            c: len(dag.nodes[c]["members"])
            for c in dag
        }

        stats = {}

        for node in dag:
            desc = nx.descendants(dag, node)
            anc = nx.ancestors(dag, node)

            # Get number of downstream / upstream nodes
            # as wins losses, use the size of any cycles
            wins = sum(sizes[d] for d in desc)
            losses = sum(sizes[a] for a in anc)

            # Same SCC as me
            num_within_scc = sizes[node] - 1

            for doc in dag.nodes[node]["members"]:
                kappa = wins + losses + num_within_scc
                stats[doc] = (wins + num_within_scc,
                              losses + num_within_scc,
                              kappa,
                              doc)

        def blitzrank_sort_key(item):
            wins, losses, kappa, doc = item
            return (losses, wins, -kappa, doc)

        top_n = sorted(stats.values(), key=blitzrank_sort_key)[:num_nodes]
        return {doc for _, _, _, doc in top_n}
