## Turning configs into jupyter notebooks

This task primarilly exists for you, the agent, to do on demand.

Turn an experiment here into a notebook in literate programming style. You MUST match the implementation here very closely. IE agentic, use agent_run, search functions here slighly modified for a notebook.

Your guidance should come from the description of the yml file. That's your lord, savior, guidance. Your gospel and source of truth.

### The task

You will be asked to take a command like

```
uv run run --strategy configs/bm25.yml --no-cache --dataset msmarco --workers 4  # run_strategy cache only
```

And create a notebook that loads msmarco, reimplements the BM25 strategy in the notebook, runs it on the dataset, and evaluates the results with NDCG/MRR.

These notebooks work as following:

### Notebook audience

The audience will be semi-aware of general search concepts and python programming.

### Notebook dependencies

* They do not depend on any code in this repo
* They're intended for a Google colab environment. So you can assume standard python data libreries like pandas, numpy, matplotlib, etc are available. But not much else.

### Mounting gdrive

This code should exist near the top to mount gdrive

```
!pip install git+https://github.com/softwaredoug/cheat-at-search.git@<commit_hash>
from cheat_at_search.data_dir import mount
mount(use_gdrive=True)    # colab, share data across notebook runs on gdrive
# mount(use_gdrive=False) # <- colab without gdrive
# mount(use_gdrive=False, manual_path="/path/to/directory")  # <- force data path to specific directory, ie you're running locally.
```

### Importing dataset + loading keys

Here's a sample cell of loading WANDS dataset. Included is the text cell above the code cell, which explains what's going on. You can assume the dataset is already indexed, so loading it is fast and doesn't require the user to run any indexing code.

```
## Get an OpenAI Key + load corpus

This will prompt you for an OpenAI Key to interact with GPT-5

-- 

from cheat_at_search.data_dir import key_for_provider
from openai import OpenAI
from cheat_at_search.wands_data import corpus, judgments

OPENAI_KEY = key_for_provider("openai")

openai = OpenAI(api_key=OPENAI_KEY)
```

### Literate programming best practices

Notebooks reimplement from a somewhat bottom-up perspective, linearly, telling a story through the notebook.

Importantly, as you implement each piece, at the end of each cell have a bit of code that demonstrates how what you created works.

Another important rule - don't introduce too much complexity for the human at once. You want to gradually build up to higher concepts, starting simple. At each step, demonstrating what you just did.

### No caching of BM25 index, results, embeddings, etc

In this repo, when run as a CLI, there's a great deal of caching to speed experimentation. Do not
attempt to do this as you'll run in a notebook environment.

The only acceptable caching is what's done when mounting above


### Top level comment - from config yaml

If the config has a `description` field, include that as a markdown cell at the top of the notebook. This gives context to the experiment being run. If not, you can write a brief description of the experiment in your own words based on the config.


### Take guidance from description as well

The intent, etc will also be in the description. Use that to document the cells you create.

### Where notebooks live

Notebooks should be placed in the `notebooks/` directory. They should be named according to the strategy and dataset they correspond to, for easy identification. For example, a notebook for the BM25 strategy on the MSMarco dataset could be named `bm25_ms_marco.ipynb`.


### Use SearchStrategy

Make sure what's central to the notebook is a SearchStrategy implementation that demonstrates teh experiment. Instantiate the strategy, call run_strategy to get results dataframes. Use cheat_at_search's ndcgs/mrrs helpers to give summary eval metrics. 


### Ignore these params

While I'll ask you to create a notebook given a config + command, you should ignore these command line params that help when running locally

--no-cache (run_strategy cache only)
--device

### Testing whether your notebooks are parsable

You should have jupyter as a dev dependency, you should test the parsinig and notebook generation with jupyter at the command line.

### Run tests

Always run the test

 uv run pytest tests/test_notebook_descriptions.py
