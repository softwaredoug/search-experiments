from cheat_at_search import esci_data, msmarco_data, wands_data

DATASETS = {
    "esci": esci_data,
    "msmarco": msmarco_data,
    "wands": wands_data,
}


def get_dataset(name: str):
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc
