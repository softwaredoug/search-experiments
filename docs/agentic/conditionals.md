## Conditionals, Constraints, Validators, and judges

Conditionals etc apply some kind of feedback to the agent after the tool calling loop is complete.

See [Agentic Strategy Docs])docs/agentic/agentic.md) for the complete details of agentic data.

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


### Jev Judge Validator

`jev_judge_relevance` is a Jev decision-model version of the LLM judge validator. It evaluates
each result for the query and assigns a label from the configured choices. Each choice maps a
label to criteria describing when that label applies:

```
    validators:
      - jev_judge_relevance:
          prompt: "Please return more relevant results to better help the user find what they're looking for."
          params:
            model: jev/jev-latest
            probability_threshold: 0.7
            confidence_threshold: 0.7
            max_runs: 2
            choices:
              Relevant: The result satisfies the user's query.
              Neutral: The result is related but does not clearly satisfy the query.
              Irrelevant: The result does not satisfy the query.
            judge_prompt: |
              Judge how well each result satisfies the query.

              Query: {query}

              Results:
              {results}
```

For each result, Jev selects a choice and returns its probability and the evaluation's
confidence. The evaluation is accepted only when both values are strictly greater than their
configured thresholds. An evaluation that does not clear either threshold is reported as
uncertain and does not count as a confidently relevant result.

The validator passes when every result has an accepted `Relevant` label. Other labels and
uncertain evaluations are included in feedback for the agent, which is asked to improve its
results and evaluated again. After `max_runs` attempts (default `2`), the validator accepts the
results to avoid an infinite loop, matching the LLM judge validator behavior.


### Jev Bag-of-Decisions Judge Validator

`jev_bag_of_decisions_judge` generates a query-specific rubric of yes/no questions with an LLM,
then evaluates every returned result against all rubric questions with Jev. Questions should be
phrased so an affirmative answer is evidence that the result satisfies the query.

```yaml
    validators:
      - jev_bag_of_decisions_judge:
          prompt: Please use this rubric feedback to improve the relevance and ordering of your results.
          params:
            model: jev/jev-latest
            positive_probability_threshold: 0.75
            negative_probability_threshold: 0.25
            max_runs: 2
            generator:
              model: gpt-5-mini
              system_prompt: |
                Generate concise yes/no questions that can help determine whether a result
                satisfies the query. An affirmative answer should indicate evidence of relevance.
              prompt: |
                Generate a varied rubric of questions for this query:

                {query}
            state_format: |
              {title}
              {description}
```

Each document's rubric score is the sum of valid `P(yes)` values, including values in the
uncertain band. Feedback only lists thresholded answers: probabilities greater than or equal to
`positive_probability_threshold` are marked `👍`, and probabilities less than or equal to
`negative_probability_threshold` are marked `👎`. Answers between those thresholds are omitted
from the criteria list. The negative threshold must be less than the positive threshold.

The rubric is generated once per query and reused during validator retries. The validator passes
when every returned result has at least one positive criterion and no negative criteria. Results
with no confident criteria, or with any negative criterion, are sent back as feedback. After
`max_runs` attempts (default `2`), the validator accepts the results to bound retries.


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

