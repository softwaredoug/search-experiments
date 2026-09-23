# Query-understanding Jev confidence thresholds

## Run

From the repository root, run:

```bash
./scripts/run_choice_single_jev_thresholds.sh
```

The script evaluates seven configurations on the full WANDS dataset: the
baseline `gpt-5-mini` choice classifier, followed by Jev at confidence
thresholds `0.6`, `0.7`, `0.8`, `0.9`, `0.95`, and `0.99`. It runs
`uv run query_classification` once per configuration, using direct category
evaluation and a ground-truth `--query-threshold` of `0.8`.

The script uses
[`configs/ecom_class/ecom_choice_single.yml`](../configs/ecom_class/ecom_choice_single.yml)
as its baseline. It generates the Jev variants in a temporary directory by
switching the model to `jev/jev-latest` and setting the confidence threshold.
The temporary configs and intermediate summary CSV are deleted when the script
exits. The base config is expected to use the `ecom_choice_single` strategy name
and `gpt-5-mini` model because the script derives the variant configs with text
substitutions.

The checked-in Jev example is
[`configs/ecom_class/ecom_choice_single_jev.yml`](../configs/ecom_class/ecom_choice_single_jev.yml).
It configures a `0.7` cutoff; the sweep generates temporary variants for the six
cutoffs listed above rather than using this file directly.

## Reading the classifier configurations

### Jev choice classifier

The following excerpt shows the relevant settings from
[`ecom_choice_single_jev.yml`](../configs/ecom_class/ecom_choice_single_jev.yml).
The full file supplies descriptions for the product-category choices and
includes `Unknown` as the abstention choice.

```yaml
strategy:
  name: ecom_choice_single_jev
  type: query_understanding
  params:
    categorize:
      field: category
      enrichment_engine:
        type: choice_single
        params:
          model: jev/jev-latest
          confidence_threshold: 0.7
          pad_missing_choices: false
          # choices: category labels/descriptions, including Unknown
          prompt: |
            Which category best describes the query?

            {query}
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title^9.4, description^4]
        boost_matches: 10
```

Jev returns one of the configured choices along with confidence. At this cutoff,
the enricher retains the category only when confidence is above `0.7`; otherwise
it returns an empty list for `Unknown`. `pad_missing_choices: false` means
choices are restricted to available corpus categories, plus the configured
`Unknown` choice. The retrieval-engine block is part of the strategy, but the
classification runner does not execute retrieval while calculating these
classification metrics.

### OpenAI GPT-5-mini baseline

The threshold script's OpenAI comparator is
[`configs/ecom_class/ecom_choice_single.yml`](../configs/ecom_class/ecom_choice_single.yml).
I could not find a `gpt-5.1-mini` classification YAML in `configs`; this
repository's matching baseline is named `gpt-5-mini` (without `.1`). The excerpt
below shows that model and its Unknown instruction; its full category choice
mapping is in the linked file.

```yaml
strategy:
  name: ecom_choice_single
  type: query_understanding
  params:
    reasoning: medium
    categorize:
      field: category
      enrichment_engine:
        type: choice_single
        params:
          model: gpt-5-mini
          pad_missing_choices: false
          # choices: same product-category labels/descriptions, including Unknown
          prompt: |
            Which category best describes the query?

            It's very important to choose "Unknown" if its unclear

            {query}
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title^9.4, description^4]
        boost_matches: 10
```

Unlike the Jev variants, this baseline has no confidence threshold. It relies on
the model's choice, with the prompt explicitly asking it to choose `Unknown`
when unclear. The threshold sweep compares this baseline with Jev's
confidence-based abstention.

The script requires the configured OpenAI and TypeSafe/Jev credentials. Each
variant classifies the full query set, so the run makes repeated model calls
across the baseline and six Jev thresholds.

## What is evaluated

This measures **query classification**, not retrieval ranking. The runner calls
the strategy's enricher and compares its category prediction with categories
derived from WANDS judgments; it does not run the configured BM25 retrieval
engine or measure NDCG/Recall@k.

For each query, ground truth is constructed from the categories of its
maximum-grade judged documents. A category is included when it accounts for at
least 80% of those documents. If no category reaches that proportion, the
ground-truth list is empty. The `--query-threshold` here is this ground-truth
category proportion; it is separate from the Jev confidence threshold.

Jev accepts a category only when its reported confidence is **strictly greater**
than the configured threshold. Confidence equal to or below the threshold, or
missing confidence, produces `Unknown`, which the enricher returns as an empty
category list. The OpenAI baseline has no confidence cutoff in this experiment.

The aggregate recall and Jaccard include all queries. With the current
classification metrics, two empty lists score 1, and exactly one empty list
scores 0. Coverage is the proportion of queries for which the enricher returns
a nonempty category list. Thus higher confidence cutoffs can reduce coverage
while changing recall among all queries.

## Generated artifacts

By default, the script overwrites:

- [`research/results/query_understanding_jev_thresholds.csv`](results/query_understanding_jev_thresholds.csv): one row per baseline/threshold variant, with `variant`, `confidence_threshold`, mean `recall`, and `coverage`.
- [`assets/query_understanding_jev_thresholds.png`](../assets/query_understanding_jev_thresholds.png): a plot of recall against coverage, labeled with the baseline or Jev confidence threshold.

The plot sorts points by coverage, connects them, and reports a trapezoidal area
over the sampled coverage range. This is not a retrieval metric or a standard
ROC AUC. The plot's y-axis is labeled “Accuracy,” while the CSV value used for
that axis is the query-classification mean recall.

The script can be configured through environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BASE_CONFIG` | `configs/ecom_class/ecom_choice_single.yml` | Baseline strategy YAML used to create the variants |
| `DATASET` | `wands` | Dataset evaluated by each run |
| `WORKERS` | `1` | Dataset loading worker count |
| `QUERY_THRESHOLD` | `0.8` | Minimum category proportion for ground truth |
| `OUTPUT_CSV` | `research/results/query_understanding_jev_thresholds.csv` | Final summary CSV path |
| `PLOT_OUTPUT` | `assets/query_understanding_jev_thresholds.png` | Plot output path |

For example, output paths can be changed without editing the script:

```bash
OUTPUT_CSV=/tmp/jev-thresholds.csv \
PLOT_OUTPUT=/tmp/jev-thresholds.png \
./scripts/run_choice_single_jev_thresholds.sh
```

Implementation references: [`scripts/run_choice_single_jev_thresholds.sh`](../scripts/run_choice_single_jev_thresholds.sh),
[`scripts/plot_choice_single_jev_thresholds.py`](../scripts/plot_choice_single_jev_thresholds.py),
and [`exps/runners/query_classification.py`](../exps/runners/query_classification.py).
