from typing_extensions import Literal

DATASET_NAMES = ("esci", "minimarco", "msmarco", "wands", "doug_blog")
DatasetName = Literal["esci", "minimarco", "msmarco", "wands", "doug_blog"]


def get_dataset(name: DatasetName, workers: int = 1, ensure_snowball: bool = True):
    from cheat_at_search.data_dir import DATA_PATH
    print(f"Loading dataset {name} from {DATA_PATH} with {workers} workers and ensure_snowball={ensure_snowball}")
    try:
        if name == "esci":
            from cheat_at_search import esci_data as dataset
        elif name == "msmarco":
            from cheat_at_search import msmarco_data as dataset
        elif name == "minimarco":
            from cheat_at_search import minimarco_data as dataset
        elif name == "wands":
            from cheat_at_search import wands_data as dataset
        elif name == "doug_blog":
            from cheat_at_search import doug_blog_data as dataset
        else:
            raise KeyError(name)
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc

    return dataset
