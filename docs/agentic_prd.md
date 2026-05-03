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

Only title and description are supported.

## Harness constraints

Some params require wrapping the agentic loop itself in a harness to drive execution.

For example, if we want to enforce a certain number of iterations, or a certain number of calls to a tool, we can do that with the harness. The harness can check the agent state after every tool call, and decide whether to continue or not.

### Stopping

One type of param is a "stopper" - when to stop the agentic loop. Even if the agent comes back, we might tell it to try again. Here we see a stopper based on required number of tool calls.

```
    stop:
      - tool_calls: 4
```

### Reprompt

If stop is not satisfied, a "reprompt" can be issued as the user message to pass to the agent as needed, ie:

```
    reprompt: You're doing really well. Please keep searching until 4 tool calls have been made so no stone is left unturned.
```
