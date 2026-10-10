This page documents teh available tools when experimenting with agentic strategies.

See [Agentic Strategy Docs])docs/agentic/agentic.md) for the complete details of agentic data.

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

Each query log includes timed start/completion/error events for model requests and tool calls,
validator execution, and per-result Jev evaluation. Active queries emit a heartbeat every 30 seconds
with their current phase and elapsed time, plus a worker stack event after two minutes. Request and
result payload contents are not added to these diagnostic events.
