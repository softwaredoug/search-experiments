# Agentic search strategies

One key type of strategy here is the 'agentic' strategy. 

Agent as in LLM style of agent.

The basic idea is to execute a set of simple search tools. Then drive an agentic loop until exhuasted. Finally
producing a ranked list of search results.

Basic example:

```yaml
strategy:
  name: agentic_bm25_ecommerce
  type: agentic
  params:
    model: gpt-5
    reasoning: medium
    system_prompt: |
      You take user search queries and use a search tool to find products.

      Look at the search tools you have, their limitations, how they work, etc when forming your plan.

      Finally return results to the user per the SearchResults schema, ranked best to worst.

      Gather results until you have 10 best matches you can find. It's important to return at least 10.

      It's very important you consider carefully the correct ranking as you'll be evaluated on
      how close that is to the average shoppers ideal ranking.
    search_tools:
      - bm25:
      - minilm:
```

This calls OpenAI with

- the system prompt here
- the query being searched for as the user prompt. 
- a set of simple search tools that the agent can call to gather info. In this case, BM25 and minilm embedding search. The agent can call these tools with different queries, etc to gather info. The agentic loop continues until the agent decides to stop (or max iterations is reached). Then the final ranked list of results is returned and evaluated.

Implementation note: agentic strategies now use `OpenAIAgent` from cheat-at-search for the tool-calling loop. The harness applies validators + stop conditions around that agent loop.

### Agent state

Everytime we start a search, we initiate "agent_state". That's like a scratchpad for the agentic loop, harness, and tools to track state and prevent illegal operations. See more in "Tool guards" below. 

## Few shop options

Options to add few-shot examples to the system prompt. These can be configured in the yml as well, and are added to the system prompt before the agentic loop starts.

It shows up as:

```
  params:
    model: gpt-5
    system_prompt: |
       ...
    few_shot:
       - sample_judgments: 10

```

### Few shot, random evaluated results

The option

```
    few_shot:
       - sample_judgments:
            num_rows: 10
``` 

Should sample the judgments to add 10 examples of queries, their products, and whether they're relevant or not.

When sampling, try to balance the number of relevant and non-relevant examples (ie exemplars of each relevance grade).

This should use the core columns expected on any data (title, description) and not other data.

To add custom columns

```
    few_shot:
       - sample_judgments:
           num_rows: 10
           columns:
            - category
            - price
```

And those would be inclulded in the prompt. If this corpus does not have this, then it should throw an error.

## Tools

The search tools here reflect specific functions executing a type of retrieval. While they share backend indices, etc with the search strategies (ie bm25 search strategy) the scoring code, etc is different.

For example, an embedding strategy would be flexible enough to let you choose any embedding model. Here we just hardcode "minilm" etc.

The following context gets passed to OpenAI from the tool:

- tool name: the function name
- tool description: a description of what the tool does, how to use it, etc. This is important for the agent to know when to call it, how to call it, etc.

Raw tools (kind: raw) are not allowed in agentic strategies. If a raw tool is listed in an agentic config, raise an error.

### Top k

To not flood the agent's context, at most the agent can request 20 results.

### Tool guards

Tools can have guards that reject calls with an error. That can be based on the parameters themselves, or the agent state.

IE here's one that rejects repeat queries too similar to previous runs:

```
      - e5_base_v2:
          guards:
            - disallow_repeated_queries
```

Notice a description of each guard gets appended to the tool description

If a tool raises an exception during a call, the agent receives a string error response instead of crashing the run.

### Dataset specific tools

Some tools can only be used for specific datasets. They leverage the structure of that corpus and otherwise should not be used.

If they're used for the wrong dataset, you should throw an error.

IE here's a BM25 tool for ESCI that also takes a "locale" parameter, which might be filtered on.

```
      - bm25_esci:
          params:
            locale: us
```

When the tool is produced internally, it should advertise the datasets it can be used for. If it can be used for any dataset, then it should return "None" etc.

By convention, the tools will have the dataset as a suffix, ie "_esci" or "_wands" etc, but that's only a convention.

### Fielded BM25

The fielded BM25 tool accepts a weighted list of fields and an operator:

```
    fields: ["title^9.3", "description^4.1"]
    operator: and
```

Operators: and, or, phrase. Phrase treats the query tokens as a single phrase and
scores the token list as one term. Only title and description are supported.

### Grep + File system tools

File system tools allow the agent to search the file system using standard commands like "ls", "cat", and "grep". It involves writing an index for the dataset on the filesystem and then giving the agent grep, ls, cat to search the file system.

See docs/agentic_filesystem_prd.md

### TODO Tool

Similar to coding agents, its useful to track todos as the agent thinks of them and to externalize cognition. But the user should explicitly request these

- todo_write: write a todo to the todo list. Takes two params, a string "todo" and a string "status" (ie "in progress", "done", "not started", or whatever you want). This should append to `agent_state["todos"]`.
- todo_read: reads the todo list and returns it as a string. This should read from `agent_state["todos"]` and return the contents as a string.

Store this on the agent_state


### Delegate task tool

A delegate_task tool gives a task to a subagent and gets results back.

The subagent gets the same tools as the calling agent, minus the delegate_task tool of course to prevent infinite delegation. The subagent also gets the task description as input which is used as user prompt.

### Agentic trace folders

Agentic runs record tool calls and outputs under a working folder rooted at:

```
~/.search-experiments/agentic/<dataset>/<strategy_name>/<timestamp>
```

This path is created via the shared run-folder utility so that run/train commands use a consistent layout.


## Harness constraints

Some params require wrapping the agentic loop itself in a harness to drive execution.

For example, if we want to enforce a certain number of iterations, or a certain number of calls to a tool, we can do that with the harness. The harness can check the agent state after every tool call, and decide whether to continue or not.

### Stopping / Validating conditionals

One type of param is a "stopper" - when to stop the agentic loop. Even if the agent comes back, we might tell it to try again. Here we see a stopper based on required number of tool calls.

```
    stop:
      - tool_calls:
          prompt: "Please make at least 4 tool calls to gather enough information before returning results."
          params:
            num_calls: 4
```

Here: params are parameters to the stopper. If the condition is not met, then `prompt` will be appended to the context
as user message and the agent called again

### Validators

Validators are just stoppers, but ALL conditions must be met. Validators should be checked in the order they're listed

```
    validators:
      - num_results:
          prompt: "Please return at least 10 results to give the user a good variety to choose from."
          params:
            min_results: 10
    stop:
      - tool_calls:
          prompt: "Please make at least 4 tool calls to gather enough information before returning results."
          params:
            num_calls: 4
```

Logically first validators are checked (in order listed). If any validator fails, its prompt is appended and the loop repeats.

Then stoppers are checked. If any stopper succeeds, the loop ends. If not, the first stopper's prompt is appended and the loop continues.

The loop also stops after a maximum number of iterations (`max_loops`, default 10) to avoid infinite retries.

### LLM judge validator

Validators can use an LLM judge to provide emoji-based feedback. Example:

```
    validators:
      - llm_judge_relevance:
          prompt: "Please return more relevant results to better help the user find what they're looking for."
          params:
            model: gpt-5-mini
            reasoning: medium
            max_runs: 2
            judge_prompt: |
              You are a helpful assistant that judges the relevance of search results to a query.

              Query: {query}

              Results:
              {results}

              Please rate the relevance of these results to the query using emojis of how well they satisfy the query.
              Allowed emojis: 😃 (relevant), 😐 (neutral), 😞 (irrelevant).

              Respond as a list of graded results with fields: emoji, title, doc_id.
```


### LLM Judge Validator

An LLM Judge validator exists to give emoji-based feedback to the agent on its performance. For example, we can give feedback on the relevance of the results returned by the agent:

```
    validators:
      - llm_judge_relevance:
          prompt: "Please return more relevant results to better help the user find what they're looking for."
          params:
            model: gpt-5-mini
            reasoning: medium
            max_runs: 2
            judge_prompt: |
              You are a helpful assistant that judges the relevance of search results to a query.

              Query: {query}

              Results:
              {results}

              Please rate the relevance of these results to the query using emojis of how well they satisfy the query.
              Allowed emojis: 😃 (relevant), 😐 (neutral), 😞 (irrelevant).
```

Above results would include title, description, and ID fields for each result.

Some validators, like this, can append to prompt the output of the process to better guide teh agent. So prompted back to the agent would be something like:

```
Please return more relevant results to better help the user find what they're looking for.

LLM evaluations:

1. 😞 Red Shoes (ID: 1234)
2. 😃 Purple Shoes
```

The judge passes when all results are 😃. If not, it retries until `max_runs` is reached,
then it accepts the results to avoid an infinite loop (default `max_runs: 2`).


### Oracle Judge Validator

One type of validator - an oracle judge.

```
    validators:
      - oracle

```

An oracle acts like an LLM judge, but labels according to the judgments of the active
dataset. Similar to LLM judge, it has the following properties:

Labels with emojis:

If a dataset has two labels, the emojis should be: [😃, 😞] 
If a dataset has three labels, the emojis should be: [😃, 😐, 😞]
If a dataset has four labels, the emojis should be: [🤩, 😃, 😐, 😞]

If a document does not have a label for a query, it should receive the most negative emoji 😞 consistent
with the rules of most open search datasets


## Plan through list of agents

A list of agents is possible

An agentic strategy might look like this:


```
  params:
    model: gpt-5
    reasoning: medium
    agents:
      planning:
        system_prompt: |
          You take user search queries and use a search tool to find products.

          Look at the search tools you have, their limitations, how they work, etc when forming your plan.

          Finally return results to the user per the SearchResults schema, ranked best to worst.

          Gather results until you have 10 best matches you can find. It's important to return at least 10.

          It's very important you consider carefully the correct ranking as you'll be evaluated on
          how close that is to the average shoppers ideal ranking.
        search_tools:
          - delegate_task:

      search:
        system_prompt: |
          You take user search queries and use a search tool to find products.

          Look at the search tools you have, their limitations, how they work, etc when forming your plan.

          Finally return results to the user per the SearchResults schema, ranked best to worst.

          Gather results until you have 10 best matches you can find. It's important to return at least 10.

          It's very important you consider carefully the correct ranking as you'll be evaluated on
          how close that is to the average shoppers ideal ranking.
        search_tools:
          - bm25:
          - minilm:

      eval:
        system_prompt: |
          You take user search queries and use a search tool to find products.

          Look at the search tools you have, their limitations, how they work, etc when forming your plan.

          Finally return results to the user per the SearchResults schema, ranked best to worst.

          Gather results until you have 10 best matches you can find. It's important to return at least 10.

          It's very important you consider carefully the correct ranking as you'll be evaluated on
          how close that is to the average shoppers ideal ranking.
        search_tools:
          - delegate_task:
    plan:
      - planning: plan how to best search for {query}
      - search: find the most relevant results for {query}
      - eval: evaluate how relevant the results are for {query}
```

Throughout this whole process, the context is identical.

However, after one agent completes, the system prompt would be patched to the next agent's system prompt, and the tools would be patched to the next agent's tools.

This means that three agents would run, with different system prompts + tools

plan dictates the order of execution of the agents. The output of one agent does not get passed as input to the next agent, but the context (including agent state) is shared across all agents. So they can communicate implicitly through that.

This is like switching between Plan <-> Build mode in coding agents. Except we're doing it sequentially.

The user prompt is specified in plan, ie above planning agent gets "plan how to best search for {query}" as user prompt, and the search agent gets "find the most relevant results for {query}" as user prompt, etc. Replacing {query} with the actual query being searched for.

Params like retrying, stopping, etc would all occur WITHIN this, so logically this is 

```
for work_item in plan:
    system_prompt = agents[agent_name].system_prompt
    user_prompt = plan[agent_name].user_prompt
    tools = agents[agent_name].tools
    while not stop_condition:
        # Call OpenAIAgent + chat
        
        if stop_condition(output):
            break
        else:
            # append stopper/validator prompt before retrying
```
