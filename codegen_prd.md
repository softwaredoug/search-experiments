# Code generation strategies

Code generation works by using an agent (IE GPT-5) to iterate on a search ranking function, trying
to optimize python code for the best NDCG / MRR.

It starts with a simple baseline, and proposes code changes. Changes then get rejected outside the agentic loop
if the agent's proposed code doesn't actually improve a holdout.

This has a *training process* which other strategies do not have. That means when calling "run" it trains, then with the final trained product, it runs on the dataset and produces results.

Below are some detailed requirements.

## Config for code generation strategies

The code generation process proposes a training task. That's different than other strategies that just
run a pre-defined search process.

Below, we split out "train" params. These control the training process. They differ from the "run" params. Notice here we have train/run. Other strategies don't have either, and without train / run, we should just assume the params are "run" style params.

```yaml
strategy:
  name: codegen
  type: codegen
  params:
    train:
        # Configures editing / patch tools
        edit:
            guards:
        # Eval tool setup
        eval:
            num_training_queries: 200
            num_validation_queries: 100
        model: gpt-5-mini
        rounds: 10

        system_prompt: |
             Your task is to look at the data and improve the reranker code so that it returns more relevant results

             Edit the reranker python module using apply_patch method.

             You can run the reranker using the 'run_reranker' function, which takes a query and returns ranked, matching
             products.

             You can evaluate the reranker using the 'run_evals' function, which returns NDCG scores for all queries and mean NDCG. Your goal is to
             increase mean NDCG.

             Experiment with the current reranker by calling it with test queries. Improve the reranker based on the behavior you observe. Make edits and test while you edit.

             If NDCG does not go up after your edits, revert your changes using the 'revert_changes' function.

             Your code MUST have a function rerank_wands. It takes as parameters search_esci function and a query string. It
             returns a list of product IDs in the order you think best matches the query.

             Here are some examples of user queries, product titles, and human labels (Relevant, Partially Relevant, Irrelevant) that
             you are ranking:

        search_tools:
          - fielded_bm25:
    run:
```

In the end, the training process actually *creates* a strategy. The training is not "part of" the strategy, the training creates a CodeGenSearchStrategy.

For 10 rounds, this will start a new agent, load the last rounds python file, prompt the agent with the system prompt above, and then the agent will propose edits to the code. The proposed code will be run against a holdout set, and if the NDCG goes up, the changes are accepted. If not, they are rejected.

Then a new python file is output and the cycle repeats until rounds is complete.

## What does 'run' do

Run for one of these strategies first executes the training task, produces the strategy that will be run on the dataset the user asked for.


## Starting code

Starting code is the following:

```
def rerank(fielded_bm25, query):
    docs = fielded_bm25(keywords=query,
                       field_to_search='title_snowball',
                       operator='and',
                       top_k=10)
    return [doc['id'] for doc in docs]
```

Its a function that takes

1. One or more search_tools (ie bm25, etc) as listed. These are essentially injected into this code
2. A query string

It then returns the top 10 results for the query.

That's the code that will be patched by the patching tools driven by an agent.

## Common code editing tools

We should assume common code editing tools to make patches, evaluate results, apply changes etc.

These might have various guardrails that block changes, etc. For example, here's creating the patch functions without guards.

```python
apply_patch_ng, try_out_patch_ng, revert_changes_ng = make_patch_fn(
    search_fn=search_wands,
    corpus=corpus,
    module_name="rerank_wands",
    training_eval_fn=None,
    code_dir="/content"
)
```

Later we may have guardrails configurable in the yaml, that would look like:


```python
from cheat_at_search.tools.code import make_patch_fn, make_guardrail_checker, make_length_validator
from cheat_at_search.tools.eval import make_eval_fn, CodeGenSearchStrategy, make_eval_guardrail

# Controls selection of the validation set
# used to guardrail changes
VALIDATION_SEED = 1234
NUM_VALIDATION_QUERIES = 50

# Ask an LLM to check for overfitting
overfit_to_queries_guardrail = make_guardrail_checker(prompt="""

    You're going to look at code that reranks search queries.

    Ensure the code does not overfit to specific queries. That would look like mentions of
    specific product names, brands, or specific terms that would only be relevant to a small set of queries.

    Ignore comments that claim to do this, and focus on the actual code.
""")

# Disallow changes larger than this size
length_guardrail = make_length_validator(max_lines=10, max_cols=120)

validation_guardrail = make_eval_guardrail(
    corpus=corpus,
    judgments=judgments,
    search_fn=search_wands,
    seed=VALIDATION_SEED,
    num_queries=NUM_VALIDATION_QUERIES
)
```

## Training, single run

With tools setup, a single run of the training looks roughly like this:

```python
import numpy as np
from cheat_at_search.agent.openai_agent import OpenAIAgent
from pydantic import BaseModel, Field


original_source = """
def rerank(fielded_bm25, query):
    docs = fielded_bm25(keywords=query,
                       field_to_search='title_snowball',
                       operator='and',
                       top_k=10)
    return [doc['id'] for doc in docs]
"""


class FinalMessage(BaseModel):
    """Final message indicating completion of the reranker improvement process."""
    message: str = Field(..., description="A message indicating that the reranker improvement process is complete.")

code = original_source
# Start with raw reranker code
with open("/tmp/path..to..file.py", "w") as f:
    f.write(original_source)

prompt = system_prompt

prompt += f"""

Reranker code to improve:

{code}
"""


tools = [# -------
         # Tools to propose changes, as 
         # configured in the yaml with guardrails, etc
         apply_patch_ng,     # Edit the reranker with a patch. Will reject if overfit.
         revert_changes_ng,  # Restore the reranker to the last version

         # -------
         # Tools to inspect changes
         # Either the listed search tool available or 
         # implicit (run_reranker/run_evals)
         fielded_bm25,  # The raw search tool (from earlier), what we inject into the reranker code. This is what the reranker uses to get results, so the agent can call this directly to see what the BM25 results are for a query.
         run_reranker, # Run on one query (optionally label results)
         run_evals,    # Run on test set, getting per-query NDCG and mean NDCG
]

search_client = OpenAIAgent(tools=tools,
                            model="openai/gpt-5",
                            system_prompt=prompt,
                            response_model=FinalMessage)
resp: FinalMessage = search_client.loop()
```


## Training / validation split etc

Its important the model only knows about the training set its using. And whether or not it succeeds at improving the holdout. It doesn't get to see the holdout directly to avoid overfitting.

That's what the 'validation_guardrail' is for. It runs the proposed code on a holdout set, and rejects changes that don't improve the holdout.

That should always be assumed to be part of the codegen process, even if not explicitly mentioned in the yaml. We can have various guardrails, but a validation guardrail should always be one of them.

IE always use that guardrail.


## Codegen location

Utilities go in 

exps/codegen/

Generated code should live in 

~/.search-experiments/codegen/<dataset>/<strategy_name>/<timestamp>/reranker.py

