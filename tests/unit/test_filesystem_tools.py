import pandas as pd
import pytest

from exps.tools.filesystem import (
    make_filesystem_cat_tool,
    make_filesystem_grep_tool,
    make_filesystem_ls_tool,
)


def _sample_corpus():
    return pd.DataFrame(
        {
            "doc_id": [101, 202, 303],
            "title": ["Red Shoes", "Ship Wheel", "Brunk Desk"],
            "description": [
                "These are the best red shoes.",
                "Decorative ship wheel for the wall.",
                "Mid-century Brunk writing desk.",
            ],
        }
    )


def test_filesystem_columns_added_once():
    corpus = _sample_corpus()
    ls_tool = make_filesystem_ls_tool(corpus)
    assert "path" in corpus.columns
    assert "contents" in corpus.columns
    assert corpus.attrs.get("_filesystem_indexed") is True

    cat_tool = make_filesystem_cat_tool(corpus)
    assert cat_tool(corpus.loc[0, "path"]).startswith("# ")

    second_tool = make_filesystem_ls_tool(corpus)
    assert second_tool("/", "**/*.txt", max_results=10)


def test_contents_format_includes_doc_id():
    corpus = _sample_corpus()
    cat_tool = make_filesystem_cat_tool(corpus)
    contents = cat_tool(corpus.loc[0, "path"])
    assert contents.startswith("# Red Shoes (ID: 101)")
    assert "These are the best red shoes." in contents


def test_ls_glob_matches_txt_files():
    corpus = _sample_corpus()
    ls_tool = make_filesystem_ls_tool(corpus)
    results = ls_tool("/", "**/*.txt", max_results=10)
    assert len(results) == 3
    assert all(result.endswith(".txt") for result in results)


def test_grep_finds_matches_case_sensitive():
    corpus = _sample_corpus()
    grep_tool = make_filesystem_grep_tool(corpus)
    results = grep_tool("Brunk", "**/*.txt", num_results=10)
    assert len(results) == 1
    assert results[0]["path"].endswith(".txt")
    assert "Brunk" in results[0]["snippet"]


def test_grep_invalid_regex_raises():
    corpus = _sample_corpus()
    grep_tool = make_filesystem_grep_tool(corpus)
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        grep_tool("[", "**/*.txt", num_results=10)


def test_ls_max_results_limit():
    corpus = _sample_corpus()
    ls_tool = make_filesystem_ls_tool(corpus)
    with pytest.raises(ValueError, match="max_results"):
        ls_tool("/", "**/*.txt", max_results=51)


def test_cat_duplicate_path_raises():
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 1],
            "title": ["Same", "Same"],
            "description": ["One", "Two"],
        }
    )
    cat_tool = make_filesystem_cat_tool(corpus)
    with pytest.raises(ValueError, match="Multiple files found"):
        cat_tool(corpus.loc[0, "path"])


def test_cat_missing_path_raises():
    corpus = _sample_corpus()
    cat_tool = make_filesystem_cat_tool(corpus)
    with pytest.raises(ValueError, match="No file found"):
        cat_tool("/missing.txt")
