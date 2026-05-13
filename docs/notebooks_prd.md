## Turning configs into jupyter notebooks

This task primarilly exists for you, the agent, to do on demand.

Turn an experiment here into a notebook in literate programming style. You MUST match the implementation here very closely. IE agentic, use agent_run, search functions here slighly modified for a notebook.

Your guidance should come from the description of the yml file. That's your lord, savior, guidance. Your gospel and source of truth.

### The task

You will be asked to take a command like

```
uv run run --strategy configs/ecom_base/bm25.yml --no-cache --dataset msmarco --workers 4  # run_strategy cache only
```

or 

```
 uv run train --strategy configs/codegen/codegen_rewrite.yml --dataset wands
```

And create a notebook that demonstrates these, step by step, for given strategy, training task, and / or dataset.

These notebooks work as following:

### Usage of OpenAI

It's assumed that OpenAI models would be called via cheat-at-search. The user will be prompted for a key by cheat-at-search when needed.

You should use the utilities in cheat-at-search that expect OpenAI and its ok for live interaction.

### Notebook audience - educating new search practioners

The audience will be semi-aware of general search concepts and python programming. 

Assume they're created for educational purposes for readers of 

### Notebook dependencies

* They do not depend on any code in this repo
* They're intended for a Google colab environment. So you can assume standard python data libreries like pandas, numpy, matplotlib, etc are available. But not much else.
* They CAN depend on cheat-at-search, as long as its tied to a commit hash. It would be the hash of this repo at the time of notebook creation. This allows you to use cheat-at-search's data loading, key management, and evaluation helpers without worrying about whether the user has the right version of cheat-at-search installed.
* They can also depend on SearchArray for full text search


### Dataset dependencies

Every dataset will have two search array columns, tokenized by snowball:

- `description_snowball` - a SearchArray column of the description
- `title_snowball` - a SearchArray column of the title, if the dataset has a title field. If not, this column will be empty or null.

You don't need to create these. Cheat at search creates them for you.

### Mounting gdrive

This code should exist near the top to mount gdrive

```
!pip install git+https://github.com/softwaredoug/cheat-at-search.git@<commit_hash>
from cheat_at_search.data_dir import mount
mount(use_gdrive=True)    # colab, share data across notebook runs on gdrive
# mount(use_gdrive=False) # <- colab without gdrive
# from pathlib import Path
# manual_path=str(Path.home() / ".search-experiments" / "cheat-at-search")
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

Importantly, not as print(sample[0].keys()), but just sample[0] to let the notebook output cells view the data.

Another important rule - don't introduce too much complexity for the human at once. You want to gradually build up to higher concepts, starting simple. At each step, demonstrating what you just did.

### Logging, etc

Whenever there's an option to pass a logger, make sure one is passed that logs to stdout.


### Top level comment - from config yaml

If the config has a `description` field, include that as a markdown cell at the top of the notebook. This gives context to the experiment being run. If not, you can write a brief description of the experiment in your own words based on the config.


### Take guidance from description as well

The intent, etc will also be in the description. Use that to document the cells you create.

### Where notebooks live

Notebooks should be placed in the `notebooks/` directory. They can be organized by a theme like "codegen" or "agentic".

They should be named according to the strategy and dataset they correspond to, for easy identification. For example, a notebook for the BM25 strategy on the MSMarco dataset could be named `bm25_ms_marco.ipynb`.


### Steps for codegen training notebook

- Coding / eval tools should be brought in from cheat-at-search
- The codegen train runner here parses yaml config, creates guardrails, tools, etc. You shouldn't do that, just put in what's needed directly in the notebook
- First demonstrate a single training run.
- Finally instantiate the training loop for some number of rounds


### Steps for 'run' commands

When recreating 'run' commands

Make sure what's central to the notebook is a SearchStrategy implementation that demonstrates teh experiment. Instantiate the strategy, call run_strategy to get results dataframes. Use cheat_at_search's ndcgs/mrrs helpers to give summary eval metrics. 


### Ignore these params

While I'll ask you to create a notebook given a config + command, you should ignore these command line params that help when running locally

--no-cache (run_strategy cache only)
--device

### Testing whether your notebooks are parsable

You should have jupyter as a dev dependency, you should test the parsinig and notebook generation with jupyter at the command line.

### Run tests

Always run the test

 uv run pytest tests/unit/test_notebook_descriptions.py

### Ignore path param

The path param helps restart runs locally, but in a notebook assume a fresh run.
