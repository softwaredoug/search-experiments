## Enrichment Engine Evaluation

For [query understanding experiments](./query_understanding.md), enrichment engines dictate how a query
is classified. The specific enrichment engines are [documented here](./enrichment_engines.md).

Recall from [query understanding docs](./query_understanding.md), we refer to the class being classified into as just "category".
The 'enrichment engine' decides how queries are resolved to categories.

The query understanding docs also describe how the classification can be studied on its own nad how
evaluation works.

## Evaluation script

Evals are run with with:

```
uv run query_classification --strategy configs/ecom_class/ecom_query_understanding.yml --dataset wands
```

This runs the enrichment engine, classifies each query into the category, then produces stats as detailed below.

## Evaluation

If the yml file has a query_understanding strategy, it will evaluate the enrichment engine by itself. as follows.

### Ground truth

We will develop a groundtruth by using the judgments for a corpus. If I certain percentage of the query <-> doc relevant results correspond to document of a category (ie a search for 'sofa' maps to 10 relevant results, and 9 are of category "Furniture"), then we can say the groundtruth for that query is "Furniture". This can be used to evaluate the enrichment engine.

This will be set with a threshold --query-threshould=0.8 (80% of positive labels must be of a category to consider that the groundtruth for that query is that category). If no category meets the threshold, then the groundtruth for that query is "Unknown".

Its possible with a low query-threshold that a query would have many ground truth categories. IE sofa might map to ['Furniture', 'Living Room']. With --query-threshold 0.4 if 50% of the relevant results are in Furniture, and 40% are in Living Room. An empty list might also be returned if no category meets the threshold.

Regardless, in the end, our ground truth is a mapping of query -> list of categories

### Eval metrics (quality)

Recall / Jaccard are computed, only applying to queries that have a non-empty prediction. Coverage is the percentage of queries that received a non-empty prediction.

Then we will measure the following per query:

- Recall: Of the expected ground truth categories, what percent were returned by the enrichment engine?
- Jaccard: The expected ground truth categories, compared to the generated list, whats the (intersection / union)?

When both category lists are empty, per-query recall and Jaccard are both 1. When exactly one list is empty, both scores are 0. Mean recall and Jaccard average only queries with a non-empty prediction: a predicted category with empty ground truth scores 0 and is included, while an empty prediction is excluded. Coverage remains the percentage of queries that received a non-empty prediction.

If a single --query is provided, we only evaluate that query, and print the expected ground truth categories, and the generated categories.

### Eval As argument

The enrichment engine's eval can be influenced with --eval-as CLI param to query_classification script.

If
--eval-as=direct we simply do a direct comparison between ground truth and predicted categories. This is the default.

However, its sometimes useful when evaluating taxonomy to focus on hierarchy with general categories being more important
to get perfect than specific categories. For example, if the ground truth is "Furniture / Living Room / Sofa" and the predicted is "Furniture / Living Room / Chair", we might want to consider that a correct prediction at the 0th level of the taxonomy, but incorrect at the 2nd level.

Hence the setting:
--eval-as=taxonomy[0]

This will evaluate the enrichment as if its a taxonomy, looking at the 0th level. Where 0 is the highest / most general.

If the category is a taxonomy, it will contain a hierarchy like "foo / bar / baz" split with a / divider. Here the 0th level is "foo", the 1st level is "bar", and the 2nd level is "baz".

If we're predicting this field - we're both predicting taxonomy, and the ground truth is taxonomy, and we evaluate only the level specified in the taxonomy.

IE if the predicted is

"foo / bar / bin"

And the ground truth is

"foo / bar / baz"

Then if EVAL_AS="taxonomy[0]", we evaluate as "foo" == "foo" -> correct

IE if the predicted is

"luz / bar / bin"

And the ground truth is

"foo / bar / baz"

Then if EVAL_AS="taxonomy[0]", we evaluate as "luz" == "foo" -> incorrect

If there is a list, we extract the 0th from each ground truth and each predicted category, and compare those lists.

In this case the query-threshold applies only to the 0th level of the taxonomy. IE a query has the following list of query <-> doc relevant results, with categories:

```
luz / bar / bin
foo / bar / baz
foo / bar / bin
lump / bar / bin
lump / bar / booz
```

Then a --query-threshold=0.4 would produce a ground truth of ['foo', 'lump'] for the 0th level of the taxonomy, and ['bar'] for the 1st level of the taxonomy.

We use that to compute recall, jaccard, etc

### Multiple eval_as

`eval_as` might be comma seperated, with many values ie --eval-as=taxonomy[0],taxonomy[1],direct

You should report these each seperately as their own stats

### Report argument

We often want to analyze the results of the enrichment evaluation.

With --report <report_file.pkl> we output a detailed report of the enrichment engine evaluation.

Recall the ground truth is built based on aggregating query <-> document relationships. With a --report=<report_file> we:

1. Take the judgments dataframe
2. Left join in every labeled document per query, with the 'category' column being labeled
3. Left join in the prediction + ground truth for this query, for each eval_as value

If eval_as is taxonomy[N] we further create a column for the Nth level of the taxonomy for both the ground truth and the predicted category.

This gets written to the report file, which can be used for further analysis.

