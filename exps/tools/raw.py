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

        ## Tokenizing terms + phrases

        Everything below assumes tokenized terms. The tokenizer can be accessed
        through corpus["description_snowball"].array.tokenizer

        As the filed implies _snowball will snowball tokenize:

        tokenizer("tokenized REd apples") -> ["token", "red", "appl"]

        In search array, search with terms by passing a tokens
        Search with phrases by passing a list of tokens

        ## Scoring functions

        ### BM25:

        - score(token): BM25 score of term
        - score([token1, token2, ...], similarity=...): BM25 score of phrase "token1 token2 ..."

        ### Direct tf stats
        - termfreqs(token): term frequency per doc
        - termfreqs([token1, token2, ...]): term frequencies for phrase "token1 token2 ..." per doc

        ### Direct df stats
        - docfreq(token): document frequency

        ### Other
        - doclengths(): document lengths
        """
        return corpus

    get_corpus.__name__ = "get_corpus"
    return get_corpus
