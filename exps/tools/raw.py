from __future__ import annotations


def make_get_corpus_tool(corpus):
    def get_corpus(agent_state=None):
        """Return the pandas DataFrame corpus.

        Columns:
        - title: title (if any) of the document
        - description: body of the document
        - title_snowball: snowball-tokenized SearchArray index for title
        - description_snowball: snowball-tokenized SearchArray index for description

        SearchArray API (lexical statistics) lives on the .array attribute of
        snowball columns, e.g. corpus["title_snowball"].array:
        - score(token, similarity=...): BM25-style scores
        - termfreqs(token): term frequency per doc
        - docfreq(token): document frequency
        - doclengths(): document lengths
        - positions(token): positions per doc
        - tokenizer: access with corpus["description_snowball"].array.tokenizer
        """
        return corpus

    get_corpus.__name__ = "get_corpus"
    return get_corpus
