from __future__ import annotations

import pytest

from exps.datasets import get_dataset


@pytest.fixture(scope="session")
def doug_blog_dataset():
    return get_dataset("doug_blog", ensure_snowball=False)


@pytest.fixture(scope="session")
def fake_wands_dataset():
    import pandas as pd
    from searcharray import SearchArray
    from cheat_at_search.tokenizers import snowball_tokenizer
    from types import SimpleNamespace

    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "title": ["Floating bed", "Platform bed", "Nightstand"],
            "description": [
                "A bed that looks like it floats.",
                "Simple bed.",
                "A small table.",
            ],
            "category": ["Furniture", "Furniture", "Bedroom"],
            "cat_subcat": [
                "Furniture / Bedroom Furniture",
                "Furniture / Bedroom Furniture",
                "Furniture / Bedroom Furniture",
            ],
            "subcategory": ["Beds", "Beds", "Nightstands"],
        }
    )
    corpus["title_snowball"] = SearchArray.index(corpus["title"], snowball_tokenizer)
    corpus["description_snowball"] = SearchArray.index(corpus["description"], snowball_tokenizer)
    judgments = pd.DataFrame(
        {
            "query_id": [1],
            "query": ["floating bed"],
            "doc_id": [1],
            "grade": [2],
        }
    )
    return SimpleNamespace(corpus=corpus, judgments=judgments)
