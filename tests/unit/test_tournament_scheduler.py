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
        "next_contest": {0, 1}
    },
    "cycle": {
        "past_contests": [ContestResult(3, 0), ContestResult(3, 1), ContestResult(3, 2),
                          # These are tied:
                          ContestResult(0, 1), ContestResult(1, 2), ContestResult(2, 0)],
        "next_contest": {3, 0}  # Smallest doc id on tied
    }
}


@w_scenarios(tournament_scenarios)
def test_tournament_scheduling(past_contests, next_contest):
    scheduler = TournamentScheduler()
    for contest in past_contests:
        scheduler.add_preference(winner=contest.winner, loser=contest.loser)
    next_contest_size = len(next_contest)
    contest = scheduler.schedule(num_nodes=next_contest_size)
    assert contest == next_contest, f"Expected {next_contest}, got {contest}"
