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
        reasoning: medium
        refresh_every: 10

        system_prompt: |
             Your task is to look at the data and improve the reranker code so that it returns more relevant results

             Edit the reranker python module using apply_patch method.

             You can run the reranker using the 'run_reranker' function, which takes a query and returns ranked, matching
             products.

             You can evaluate the reranker using the 'run_evals' function, which returns NDCG scores for all queries and mean NDCG. Your goal is to
             increase mean NDCG.

             Experiment with the current reranker by calling it with test queries. Improve the reranker based on the behavior you observe. Make edits and test while you edit.

             If NDCG does not go up after your edits, revert your changes using the 'revert_changes' function.

Your code MUST have a function rerank_wands. It takes the query first, followed by search tools, then **kwargs,
and returns a list of product IDs in the order you think best matches the query.

             Here are some examples of user queries, product titles, and human labels (Relevant, Partially Relevant, Irrelevant) that
             you are ranking:

        search_tools:
          - fielded_bm25:
    run:
```

In the end, the training process actually *creates* a strategy. The training is not "part of" the strategy, the training creates a CodeGenSearchStrategy.

For N rounds (from --rounds), this will start a new agent, load the last rounds python file, prompt the agent with the system prompt above, and then the agent will propose edits to the code. The proposed code will be run against a holdout set, and if the NDCG goes up, the changes are accepted. If not, they are rejected.

Then a new python file is output and the cycle repeats until rounds is complete.

## What does 'run' do

Run for one of these strategies first executes the training task, produces the strategy that will be run on the dataset the user asked for.


## Starting code

Starting code is the following:

```
def rerank(query, fielded_bm25, **kwargs):
    docs = fielded_bm25(
        keywords=query,
        fields=['title^9.3', 'description^4.1'],
        operator='and',
        top_k=10,
    )
    return [doc['id'] for doc in docs]
```

Its a function that takes

1. A query string (first)
2. One or more search_tools (ie bm25, etc) as listed
3. **kwargs for runtime-only args (ignore unless instructed)

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
def rerank(query, fielded_bm25):
    docs = fielded_bm25(
        keywords=query,
        fields=['title^9.3', 'description^4.1'],
        operator='and',
        top_k=10,
    )
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

Validation is now explicit: include `validation` in `edit.guards` when you want the guardrail applied. Guardrails are only enforced when listed.


## Codegen location

Utilities go in 

exps/codegen/

Generated code should live in 

~/.search-experiments/codegen/<dataset>/<strategy_name>/<timestamp>/reranker.py


## Training continuation

Training is run via `uv run train` with `--continue`.

Right now, this is implemented for codegen only, and anything else should throw an error.

When this is set, you find the latest codegen run of the current dataset and use the last rerank function
there to continue training.

You copy the artifacts over to a new run and continue training.

IE we're training wands. If this is the latest ~/.search-experiments/codegen/wands/codegen_sample/20260427_181259/                                  

The last round there is rerank_round_4.py

The config tells you to run 10 rounds. So you keep going 10 more rounds, up to 14

You copy rounds.jsonl from the original run, and continue to append to it (so now we should have 14)

If a path is provided to --continue, don't use the latest, use the provided path. This allows users to continue from any point, not just the latest.


## Training rounds

The command line argument --rounds (e.g. `--rounds 10`) controls the number of rounds and is intended to be used instead of editing configs.


## Refreshing training sets

Use train.refresh_every to control how often codegen regenerates tools and resamples
the training/validation query sets. The default is refresh_every == rounds (once at
the start). Set refresh_every to 1 to refresh every round.


## Trained search strategy as tool 

For data hiding purposes, and to help focus on an important part of the task. 

To do this, it can be useful to use a past iterations trained strategy as a tool. IE:


```yaml
        search_tools:
          - codegen:
              path: ~/.search-experiments/codegen/wands/codegen_sample/20260427_181259/
              name: search
              description: |
                Search the WANDS dataset and return results.
              dependencies:
                - fielded_bm25
              return_fields:
                - category
```

(This is a tool either usable in an agentic strategy or to be taken as an argument to codegen).

name / description optional. If not provided:

- name: search
- description: Search the dataset and return results.

### Notice the dependencies

It's expected this creates a search function that takes a query and top_k (max 20).
It must return results with id, title, description, and score. If return_fields are provided,
include those columns in the response as well.

It would be instantiated with the other agentic tools, in a factory for making tools from a config, ie

```
def make_codegen_tool(
    corpus,
    dependencies: list,
    path: str,
    name: str,
    description: str,
    return_fields: list | None = None,
):
    # Load the last round's reranker from the provided path
    last_reranker_path = find_last_reranker(path)
    reranker_fn = load_reranker(last_reranker_path)

    # Create a tool that wraps the reranker function, injecting dependencies as needed
    def search(query, top_k=10, **kwargs):
        """Search the corpus, return top results."""
        # Inject dependencies into the reranker function as needed
        return reranker_fn(query=query, **dependencies, **kwargs)

    # Patch the tool name / description as needed
    # ...
    return search 
```

Now "search" here (or really named the "name" param) should be available to either:

1. An agentic strategy as a tool
2. A codegen strategy as a tool (for training a new codegen strategy on top of the last one's results)

### Notice the return fields

The return fields is a list of fields the tool returns for each doc. These correspond to pandas columns in the corpus that should also be returned. On top of the default title, description, doc_id, and score


### This can be deeply layered

Of course, we could have something layered like.

So you'll need to plan for stuff like this


```yaml
        search_tools:
          - codegen:
              path: ~/.search-experiments/codegen/wands/codegen_sample/20260426_121212/
              name: search
              dependencies:
                - codegen:
                    path: ~/.search-experiments/codegen/wands/codegen_sample/20260427_181259/
                    name: search
                    dependencies:
                      - fielded_bm25
```

## Raw search tools

Raw search tools work differently than the tools so far. They can only be used in codegen. An agent cannot call them as a 'tool'.

```
    search_tools:
      - raw:
        - get_corpus
```

In this case, we should not list the tool with the other tools.

Instead the name + description should be appended to the system prompt to describe them, ie

```
<main system prompt>

## Additionally injected search code

The following functions are available to generated code, injected into the rerank function:

### Tool Name

< Tool description>
```


## Starting Code

The config might specify the code to start with directly.

If specified w/ continue passed during training, then you ignore the stort code.

This would look like this in the training section of param:

```yaml
strategy:
  name: codegen
  type: codegen
  params:
    train:
      start_code: |
        def rerank(query, fielded_bm25, **kwargs):
            docs = fielded_bm25(
                keywords=query,
                fields=['title^9.3', 'description^4.1'],
                operator='and',
                top_k=10,
            )
            return [doc['id'] for doc in docs]
```

If the tools differ from what's listed, then you should throw an error. 

The simplest way to test is to simply try to run the starting code with the provided tools, and if it throws an error, then the tools don't match the code and you should throw an error that the start function has a mismatch.
