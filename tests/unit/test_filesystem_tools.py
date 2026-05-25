import pandas as pd
import pytest

from exps.tools.filesystem import (
    make_filesystem_cat_tool,
    make_filesystem_grep_tool,
    make_filesystem_ls_tool,
    make_filesystem_ls_wands_tool,
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
    assert corpus.attrs.get("_filesystem_indexed") == "default"

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
    result = grep_tool("[", "**/*.txt", num_results=10)
    assert isinstance(result, str)
    assert "Invalid regex pattern" in result


def test_grep_truncates_with_message():
    corpus = _sample_corpus()
    grep_tool = make_filesystem_grep_tool(corpus)
    result = grep_tool(".", "**/*.txt", num_results=2)
    assert isinstance(result, list)
    assert result[-1]["snippet"].startswith("Truncated (")


def test_ls_max_results_limit():
    corpus = _sample_corpus()
    ls_tool = make_filesystem_ls_tool(corpus)
    result = ls_tool("/", "**/*.txt", max_results=51)
    assert isinstance(result, list)
    assert len(result) == 3


def test_cat_duplicate_path_raises():
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 1],
            "title": ["Same", "Same"],
            "description": ["One", "Two"],
        }
    )
    cat_tool = make_filesystem_cat_tool(corpus)
    result = cat_tool(corpus.loc[0, "path"])
    assert isinstance(result, str)
    assert "Multiple files found" in result


def test_wands_path_structure():
    corpus = pd.DataFrame(
        {
            "doc_id": [101, 202, 303],
            "title": ["Red Shoes", "Ship Wheel", "Brunk Desk"],
            "description": ["One", "Two", "Three"],
            "category": ["Decor & Pillows", "Outdoor", ""],
            "subcategory": ["Wall Decor", "", ""],
        }
    )
    ls_tool = make_filesystem_ls_wands_tool(corpus)
    paths = ls_tool("/", "**/*.txt", max_results=10)
    assert "/decor-pillows/wall-decor/red-shoes-101.txt" in paths
    assert "/outdoor/ship-wheel-202.txt" in paths
    assert "/brunk-desk-303.txt" in paths


def test_wands_ls_root_lists_categories():
    corpus = pd.DataFrame(
        {
            "doc_id": [101, 202, 303],
            "title": ["Red Shoes", "Ship Wheel", "Brunk Desk"],
            "description": ["One", "Two", "Three"],
            "category": ["Decor & Pillows", "Outdoor", ""],
            "subcategory": ["Wall Decor", "", ""],
        }
    )
    ls_tool = make_filesystem_ls_wands_tool(corpus)
    results = ls_tool("/", "*", max_results=10)
    assert "/decor-pillows" in results
    assert "/outdoor" in results


def test_wands_ls_root_orders_dirs_before_files():
    corpus = pd.DataFrame(
        {
            "doc_id": [101, 202, 303],
            "title": ["Red Shoes", "Ship Wheel", "Brunk Desk"],
            "description": ["One", "Two", "Three"],
            "category": ["Decor & Pillows", "Outdoor", ""],
            "subcategory": ["Wall Decor", "", ""],
        }
    )
    ls_tool = make_filesystem_ls_wands_tool(corpus)
    results = ls_tool("/", "*", max_results=10)
    dirs = [path for path in results if path.count("/") == 1 and not path.endswith(".txt")]
    files = [path for path in results if path.endswith(".txt")]
    assert dirs
    assert files
    assert results.index(dirs[-1]) < results.index(files[0])


def test_wands_nested_glob_matches():
    corpus = pd.DataFrame(
        {
            "doc_id": [101, 202],
            "title": ["Red Rug", "Blue Rug"],
            "description": ["One", "Two"],
            "category": ["Rugs", "Rugs"],
            "subcategory": ["Outdoor", "Indoor"],
        }
    )
    ls_tool = make_filesystem_ls_wands_tool(corpus)
    results = ls_tool("/", "/rugs/**/*.txt", max_results=10)
    assert "/rugs/outdoor/red-rug-101.txt" in results
    assert "/rugs/indoor/blue-rug-202.txt" in results


def test_wands_exact_depth_glob_matches():
    corpus = pd.DataFrame(
        {
            "doc_id": [101, 202, 303],
            "title": ["Red Rug", "Blue Rug", "Green Rug"],
            "description": ["One", "Two", "Three"],
            "category": ["Rugs", "Rugs", "Rugs"],
            "subcategory": ["Outdoor", "", "Indoor"],
        }
    )
    ls_tool = make_filesystem_ls_wands_tool(corpus)
    results = ls_tool("/", "/*/*/*.txt", max_results=10)
    assert "/rugs/outdoor/red-rug-101.txt" in results
    assert "/rugs/indoor/green-rug-303.txt" in results
    assert "/rugs/blue-rug-202.txt" not in results


def test_wands_tools_require_wands_indexing():
    corpus = _sample_corpus()
    make_filesystem_ls_tool(corpus)
    with pytest.raises(ValueError, match="Filesystem already indexed"):
        make_filesystem_ls_wands_tool(corpus)


def test_cat_missing_path_raises():
    corpus = _sample_corpus()
    cat_tool = make_filesystem_cat_tool(corpus)
    result = cat_tool("/missing.txt")
    assert isinstance(result, str)
    assert "No file found" in result


def test_cat_normalizes_relative_path():
    corpus = _sample_corpus()
    cat_tool = make_filesystem_cat_tool(corpus)
    path = corpus.loc[0, "path"].lstrip("/")
    contents = cat_tool(path)
    assert contents.startswith("# Red Shoes (ID: 101)")


def test_cat_filename_only_match():
    corpus = _sample_corpus()
    cat_tool = make_filesystem_cat_tool(corpus)
    filename = corpus.loc[1, "path"].split("/")[-1]
    contents = cat_tool(filename)
    assert contents.startswith("# Ship Wheel (ID: 202)")


def test_long_title_truncates_filename():
    long_title = "x" * 400
    corpus = pd.DataFrame(
        {
            "doc_id": [12345],
            "title": [long_title],
            "description": ["desc"],
        }
    )
    ls_tool = make_filesystem_ls_tool(corpus)
    paths = ls_tool("/", "**/*.txt", max_results=5)
    assert len(paths) == 1
    path = paths[0]
    filename = path.lstrip("/")
    assert len(filename) <= 200
    assert "12345" in filename
    assert filename.endswith(".txt")
