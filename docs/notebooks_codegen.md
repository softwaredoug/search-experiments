# Guidance for codegen notebooks

This is guidance for transforming a yml for strategy:codegen into a notebook.

## General notebook guidelines

See docs/notebooks_prd.md for general guidelines on how to structure and write notebooks.

## Codeegen specific guidance

- Coding / eval tools should be brought in from cheat-at-search
- The codegen train runner here parses yaml config, creates guardrails, tools, etc. You shouldn't do that, just put in what's needed directly in the notebook
- First demonstrate a single training run.
- Finally instantiate the training loop for some number of rounds


### Incorporating tools

Non coding, non eval tools (ie tools that actually search) need to be recreated in the notebook.

When brining tools from the codebase into this repo, some important requirements must be satisfied

First the type annotations of the tools must be respected. These are interpreted into a schema to be enforced when the agent calls the tool. If the agent calls the tool with the wrong params, an error will be thrown. This is important to prevent the agent from calling tools in ways that don't make sense, and to help guide it towards calling tools correctly.

Second, the name and description of the tool must be respected, as they are important for the agent to know when and how to call the tool.


### Testing

In addition to normal notebook testing:

As usual, test down to the occurence of agent running. Then look at output cells for error.

If you see the agent begin to run successfull (output looks like work is being done) you can consider it a success
