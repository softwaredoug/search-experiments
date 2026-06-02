# End-to-end tests

End to end tests are tests that test this codebase end-to-end. They mock at 
library / API boundaries, but do not mock internals of this code.

They exist in tests/e2e/

## Test structure

See [runner tests](docs/runner_tests_prd.md) for context. 

Like these tests, the tests

- Use an inline yaml in the test that defines the strategy
- Runs the strategy
- Confirms behavior

Unlike these tests, we:

- Mock calls to agents (ie mock the OpenAIAgent)
- Use the smaller dougs_blog dataset
- Create mock embeddings where possible

## OpenAI Agent Mocked

OpenAIAgent should be mocked. Its main job is to run the agent tool-calling loop.

With its loops it can simulate tool calls and updates to context, and outputs, needed to run.

When mocking agents, use an explicit scripted tool-call sequence to make behaviors deterministic and testable. This lets us:

- Pin exact tool call order and parameters (including missing/invalid calls).
- Emit tailored model outputs per step (ranked results, categories, errors).
- Validate tool availability by name (fail fast when configs are wrong).
- Test agent behaviors with or without tool calls (empty script vs. tool call + output).
- Drive multi-agent flows (select/scatter/gather) with different outputs per agent.
- Separate tool execution from final model output (simulate tool success + model response).

## Embeddings mocked

For all embeddings below, assume a 3D vector

Mocked embeddings should be their own test utility. We will, for each document

1. Get the right passage_fn, as in this repo (from `embedding_utils.py::max_passage_fn`)
2. Generate the document text from this function
3. Use that as the key in dictionary of `vectors` which we'll "encode" as per below

### How do we encode?

1. Enumerate every query in the dataset
2. Tokenize into terms
3. Assign each term a random 3D vector if it hasn't received one already

Save a query vector as the average of all term vectors:

```
query_vectors[query] = sum(term_vector for term in query_terms) / len(query_terms)
```

Initialize every document to a random vector with a low weight:

```
doc_vectors = {}
doc_vectors[doc_text] = [(0.1, random_vector)]
```

Now for every query term, will append their vectors to a dictionary holding vectors and a weight, ie

```
doc_vectors[doc_text].append((weight, query_term_vector))
```

In the end, we will compute each documents vector:

```
final_vector = sum(weight * query_term_vector for weight, query_term_vector in vectors[doc_text])
vectors[doc_text] = final_vector
```

Now these vectors are used in a mock of load_or_create_embeddings from the cheat-at-search embeddings model.


### Filesystem

Use a temp directory to patch any filesystem calls as needed


### SearchArray / lexical search

Any lexical search tools can just be used directly
