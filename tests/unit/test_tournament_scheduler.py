from exps.judge.tournament import TournamentScheduler
from typing import NamedTuple, Any, Dict, Sequence, cast

import pytest


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    """Decorate for parametrizing tests that names the scenarios and params."""
    return pytest.mark.parametrize(
        cast(Sequence[str], [key for key in scenarios.values()][0].keys()),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys())
    )


class ContestResult(NamedTuple):
    winner: int
    loser: int


tournament_scenarios = {
    "simple": {
        "past_contests": [ContestResult(0, 1), ContestResult(1, 2)],
        "sample": {0, 1, 2, 3, 4},
        "n_per_round": 2,
        "next_contest": {3, 4}
    },
    "cycle": {
        "past_contests": [ContestResult(3, 0), ContestResult(3, 1), ContestResult(3, 2),
                          # These are tied:
                          ContestResult(0, 1), ContestResult(1, 0)],
        "sample": {0, 1, 2, 3},
        "n_per_round": 2,
        "next_contest": {2, 0}  # Smallest doc id on tied
    },
    "no_preferences": {
        "past_contests": [],
        "sample": {0, 1, 2, 3},
        "n_per_round": 2,
        "next_contest": {0, 1}
    },
    "sparse_preferences_with_unseen_ids": {
        "past_contests": [ContestResult(0, 1), ContestResult(1, 2)],
        "sample": {0, 1, 2, 3, 4},
        "n_per_round": 2,
        "next_contest": {3, 4}
    },
    "nonconsecutive_unseen_ids": {
        "past_contests": [ContestResult(4, 9)],
        "sample": {4, 6, 9, 12},
        "n_per_round": 2,
        "next_contest": {6, 12}
    },
    "cycle_with_unseen_ids": {
        "past_contests": [ContestResult(3, 0), ContestResult(3, 1), ContestResult(3, 2),
                          ContestResult(0, 1), ContestResult(1, 2), ContestResult(2, 0)],
        "sample": {0, 1, 2, 3, 4, 5},
        "n_per_round": 3,
        "next_contest": {3, 4, 5}
    },
    "resolved_winner_is_not_scheduled": {
        "past_contests": [
            ContestResult(0, 1), ContestResult(0, 2), ContestResult(0, 3)
        ],
        "sample": {0, 1, 2, 3},
        "n_per_round": 2,
        "next_contest": {1, 2}
    },
    "one_representative_per_unresolved_scc": {
        "past_contests": [
            ContestResult(0, 1), ContestResult(1, 2), ContestResult(2, 0),
            ContestResult(4, 3)
        ],
        "sample": {0, 1, 2, 3, 4},
        "n_per_round": 3,
        "next_contest": {0, 3, 4}
    },
    "all_resolved_returns_empty_selection": {
        "past_contests": [
            ContestResult(0, 1), ContestResult(1, 2), ContestResult(2, 3)
        ],
        "sample": {0, 1, 2, 3},
        "n_per_round": 2,
        "next_contest": set()
    },
    "scc_members_do_not_inflate_external_reach": {
        "past_contests": [ContestResult(0, 1), ContestResult(1, 0)],
        "sample": {0, 1, 2, 3},
        "n_per_round": 2,
        "next_contest": {0, 2}
    },
}


@w_scenarios(tournament_scenarios)
def test_tournament_scheduling(past_contests, sample, n_per_round, next_contest):
    scheduler = TournamentScheduler(all_nodes=sample, n_per_round=n_per_round)
    for contest in past_contests:
        scheduler.add_preference(winner=contest.winner, loser=contest.loser)
    contest = scheduler.schedule()
    assert contest == next_contest, f"Expected {next_contest}, got {contest}"
