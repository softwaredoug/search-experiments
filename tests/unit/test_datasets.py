import pytest

from exps.datasets import DATASET_NAMES, get_dataset


@pytest.mark.parametrize(
    ("name", "module_name"),
    [
        ("scifact", "cheat_at_search.scifact_data"),
        ("trec_covid", "cheat_at_search.trec_covid_data"),
    ],
)
def test_dataset_is_registered(name, module_name):
    dataset = get_dataset(name)

    assert name in DATASET_NAMES
    assert dataset.__name__ == module_name
