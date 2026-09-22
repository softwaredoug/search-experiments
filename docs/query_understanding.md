# Query understanding strategy

In query understanding strategy, for a given dataset, we assume the corpus has a specific column that 
would be ideal to filter to. I'll generally refer to "category" here as that column, though this should be configured.

The goal is to use an LLM or other approach prior to searching, to identify the best category for the query, then return
the search results filtered or boosting results from that category (depending on configuration)

This strategy focuses on manageable vocabulary sizes, like dozens. Not 1000s.

Basic example

```yaml
strategy:
  name: query_understanding_category
  type: query_understanding
  params:
    reasoning: medium
    categorize:
      field: category
      enrichment_engine:
        type: llm_single
        model: gpt-5
        params:
          prompt: |
            For the given query, please generate the appropriate {field} to filter to zero-in on
            the most relevant results.

            If you're unsure, or the query is ambiguous, return "Unknown" as the category.

            Here's the query:

            {query}
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title^9.4, description^4]  # Baseline BM25
        boost_matches: 10  # How much to boost results that match the category returned by the enrichment engine
         
```

## Enrichment engine

The 'enrichment engine' decides how queries are resolved to categories.

We should assume all enrichers take the query and return a list. Even the "single" ones. This just makes it easier
to handle the output.

### Dummy enricher

The dummy enricher always takes the query and returns the first category from the vocabulary.

### LLM enrichment single (llm_single)

The LLM enrichment engine implemented using [cheat at search's AutoEnricher](https://github.com/softwaredoug/cheat-at-search/blob/main/cheat_at_search/enrich/enrich.py)

It looks over the unique values of the field in the corpus. IE unique `category` values available. From these
unique values a Literal[...] is created.

In this literal an optional 'Unknown' category is added to the list of categories. This allows the LLM to return a category that is not in the corpus, if it is unsure.

Then it creates a structured pydantic BaseModel instance with the field to categorize:

BaseModel:

```python
class CategoryEnrichment(BaseModel):
    """A search query classified to the appropriate category for filtering or boosting results"""
    category: Literal[(categories list)] = Field(
        ..., description="The category to filter to for the given query"
    )
```

Replace category with the field specified for the config (ie `field:color` would look roughly like):

```python
class ColorEnrichment(BaseModel):
    """A search query classified to the appropriate color for filtering or boosting results"""
    color: Literal[(colors list)] = Field(
        ..., description="The color to filter to for the given query"
    )
```

It's then run with something like this 

```python
enricher = AutoEnricher(
     model="gpt-5-mini",
     system_prompt="You are a helpful furniture shopping agent that helps users construct search queries.",
     response_model=QueryClassification
)

def build_prompt(query, field):
    return prompt.format(field=field, query=query)

def fully_classified(query, field):
    prompt = get_prompt_fully_qualified(query, field)
    classification = enricher.enrich(prompt).model_copy()
    if "No Classification Fits" in classification.classifications:
        classification.classifications = []
    return classification

(of course replaced with whatever class structure you create)
```


### LLM enrichment multiple (llm_multiple)

Just like 'single' enricher, except we allow a list of possible categories.

This allows multiple categories to be returned. Or none if none match. IE it should produce structured
output like this:

```python
class CategoryEnrichmentMultiple(BaseModel):
    """A search query classified to the appropriate category for filtering or boosting results"""
    categories: List[Literal[(categories list)]] = Field(
        ..., description="The categories to filter to for the given query"
    )
```

The configuration is the same as `llm_single`, but uses `llm_multiple`:

```yaml
enrichment_engine:
  type: llm_multiple
  params:
    prompt: |
      For the given query, please generate the appropriate {field} values.
      If you're unsure, return an empty list.

      Here's the query:

      {query}
```


### Described choice enrichment single (choice_single)

A choice, but with a description.

Each choice has a labeled criteria for why it would be chosen. This is useful for a small vocabulary of categories, where we can describe each category to the LLM.

```yaml
enrichment_engine:
  type: choice_single
  params:
    model: gpt-5-mini
    pad_missing_choices: False
    choices:
      Furniture: A search for products used to make a room suitable for living or working, such as chairs, tables, and beds.
      Home Improvement: A search for products and services that help improve the functionality, aesthetics, or value of a home, such as tools, paint, and renovation services.
      Décor & Pillows: A search for products used to enhance the aesthetic appeal of a space, including decorative items, pillows, and other accessories.
      Outdoor: A search for products designed for use outside, such as patio furniture, gardening tools, and outdoor lighting.
      Storage & Organization: A search for products that help organize and store items, including shelves, bins, and closet organizers.
      Lighting: A search for products that provide illumination, including lamps, light fixtures, and bulbs.
      Rugs: A search for products used to cover and decorate floors, including area rugs, runners, and mats.
      Bed & Bath: A search for products related to bedrooms and bathrooms, including bedding, towels, and bathroom accessories.
      Kitchen & Tabletop: A search for products related to kitchens and dining, including cookware, utensils, and tableware.
      Baby & Kids: A search for products designed for infants and children, including toys, clothing, and nursery furniture.
      School Furniture and Supplies: A search for products used in educational settings, including desks, chairs, and school supplies.
      Appliances: A search for products that are electrical or mechanical machines designed to perform specific household tasks, such as refrigerators, washing machines, and microwaves.
      Holiday Décor: A search for seasonal decorations and accessories used to celebrate holidays.
      Unknown: No other category applies
    prompt: |
      Which best describes the query?
      {query}
```

#### Behavior in OpenAI

`choices` is a YAML dictionary mapping each choice name to its description. It should not be a block-scalar string.

If openai/* or just gpt-* model is specified:

- In OpenAI, the choices are passed to the model at the end of the prompt
- The keys of legal categories are used are Literal values in the Pydantic structured outputs
- If `pad_missing_choices` is True, any key does not have a choice description / is not in choices dictionary in the yaml, we still pass it as a Literal value, but its criteria is not listed in the prompt itself. Otherwise they're not classified into.


#### Beharior in Jev

If model: jev/* model is specified:

Review [Jev's documentation](https://jev.pro/primitives/choice/) it's a new approach to classification

- The prompt becomes the "instruction"
- Each choice becomes a jev "criteria" dictionary
- Missing choices are passed as keys, with empty descriptions depending on `pad_missing_choices`

## Enrichment ground truth and evaluation

A script exists to evaluate the enrichment engine against a corpus with judgments.

```
uv run query_classification --strategy configs/ecom_class/ecom_query_understanding.yml --dataset wands
```

If the yml file has a query_understanding strategy, it will evaluate the enrichment engine by itself. as follows.

We will develop a groundtruth by using the judgments for a corpus. If I certain percentage of the query <-> doc relevant results correspond to document of a category (ie a search for 'sofa' maps to 10 relevant results, and 9 are of category "Furniture"), then we can say the groundtruth for that query is "Furniture". This can be used to evaluate the enrichment engine.

This will be set with a threshold --query-threshould=0.8 (80% of positive labels must be of a category to consider that the groundtruth for that query is that category). If no category meets the threshold, then the groundtruth for that query is "Unknown".

Its possible with a low query-threshold that a query would have many ground truth categories. IE sofa might map to ['Furniture', 'Living Room']. With --query-threshold 0.4 if 50% of the relevant results are in Furniture, and 40% are in Living Room. An empty list might also be returned if no category meets the threshold.

Regardless, in the end, our ground truth is a mapping of query -> list of categories

Then we will measure the following per query that receives a prediction:

- Recall: Of the expected ground truth categories, what percent were returned by the enrichment engine?
- Jaccard: The expected ground truth categories, compared to the generated list, whats the (intersection / union)?

We summarize then average per-query recall, average per-queryi jaccard, and coverage: percentage of queries taht received a non-empty prediction

If a single --query is provided, we only evaluate that query, and print the expected ground truth categories, and the generated categories.


### Eval As environment variable

The enrichment engine's eval can be influenced with --eval-as CLI param.

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


## Retrieval engine

The retrieval engine then decides what to do with the output of the enrichment engine. Below
details the options for BM25 enrichment

### bm25_boosted 

Run a BM25 search over the corpus (see the bm25 strategy for details) and boost results that match the category returned by the enrichment engine.

The ^ operator is used as a weight on the title / description fields.

Then, within these results, search the category of the query by phrase. A rough

```python
# ****
# If there's a category, boost that by a constant amount
for category in classified.categories:
    tokenized_category = snowball_tokenizer(category)
    category_match = np.ones(len(self.index))
    if tokenized_category:
        category_match = self.index['category_snowball'].array.score(tokenized_category) > 0
    bm25_scores[category_match] += self.category_boost
```


### bm25_hierarchy_boosted : Hierarchy boosted

Split the category into a hierarchy using the / separator. Then boost results that match the category at each level of the hierarchy, with a decaying boost for each level.

A search for 

"foo / bar / baz" would boost results that match "foo" the most, then "foo / bar" less, then "foo / bar / baz" the least.

Since we might have multiple classifications for the queries, we do this for each classification, and sum the boosts.

The configuration uses `boost_matches` for the level-zero boost and `decay` for
the multiplier applied at each subsequent level. For example, with
`boost_matches: 10` and `decay: 0.5`, the boosts for `foo / bar / baz` are
`10`, `5`, and `2.5`.

```yaml
retrieval_engine:
  base: bm25_hierarchy_boosted
  params:
    fields: [title^9.4, description^4]
    boost_matches: 10
    decay: 0.5
```
