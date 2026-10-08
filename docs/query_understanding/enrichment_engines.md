## Enrichment engines

For [query understanding experiments](./query_understanding.md), enrichment engines dictate how a query
is classified.

Recall from [query understanding docs](./query_understanding.md), we refer to the class being classified into as just "category".
The 'enrichment engine' decides how queries are resolved to categories.

The query understanding docs also describe how the classification. How evalution works is fonud in [enrichment_engines_eval](./enrichment_engines_eval.md).

It *is* the classifier. Below is a list of enrichment engines.

### Dummy enricher

The dummy enricher always takes the query and returns the first category from the vocabulary. Used primarilly
for e2e testing.


### Oracle enricher

The oracle enricher loads the judgments and creates a list of relevant doc_ids (all results where the grade is max_grade) for the query.

Then it returns the list of categories for all doc_ids from the corpus.

Empty categories are removed.

```yaml
enrichment_engine:
  type: oracle
```

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


### LLM choice (`llm_choice`)

A single-choice LLM enricher. It uses structured output to select from a finite
vocabulary, with optional descriptions that are included in the prompt. This
engine accepts unprefixed/OpenAI models only; Jev models use `jev_choice_single`.

Each choice has a labeled criteria for why it would be chosen. This is useful for a small vocabulary of categories, where we can describe each category to the LLM. Choices are optional.

```yaml
enrichment_engine:
  type: llm_choice
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


### Jev choice single (`jev_choice_single`)

Uses Jev's Choice primitive to select one category. Each choice name is passed
as a criterion label and its optional description is passed as the criterion.
This engine accepts Jev models only.

```yaml
enrichment_engine:
  type: jev_choice_single
  params:
    model: jev/jev-latest
    confidence_threshold: 0.7
    choices:
      Furniture: Products used to furnish a room.
      Lighting: Products that provide illumination.
    prompt: |
      Which category best describes the query?
      {query}
```

### Jev choice multiple (`jev_choice_multiple`)

Uses Jev's Choice probabilities to return every category whose probability is
strictly greater than the configured threshold. It returns an empty list when
no category exceeds the threshold. This engine accepts Jev models only.

```yaml
enrichment_engine:
  type: jev_choice_multiple
  params:
    model: jev/jev-latest
    pad_missing_choices: False
    threshold: 0.4
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


### Jev BM25 then select

Both engines perform an independent BM25 search to build per-query category
candidates. The configured query-understanding retrieval engine then runs a
second search using the selected category or categories.

Each engine takes the top `aggregate_over` positive-score matches, counts
their category values, and offers the most popular 255 categories to Jev.
Documents with a score of zero are not matches. If there are no positive-score
matches or no category values in those matches, enrichment returns an empty list.

Optional `retrieval.k1` and `retrieval.b` settings configure candidate-search
BM25 scoring; they default to `1.2` and `0.75`.

#### Single (`jev_bm25_then_select`)

Jev's selection is returned only when confidence is strictly greater than
`confidence_threshold`.

```yaml
enrichment_engine:
  type: jev_bm25_then_select
  params:
    model: jev/jev-latest
    confidence_threshold: 0.7
    aggregate_over: 1000
    prompt: |
      Which category best describes the query?
      {query}
    retrieval:
      fields: [title^9.4, description^4]  # Candidate-search BM25 fields/weights
```

#### Multiple (`jev_bm25_then_select_multiple`)

Returns every candidate category whose Jev Choice probability is strictly
greater than `threshold`. No `Unknown` option is added; if no category exceeds
the threshold, enrichment returns an empty list.

```yaml
enrichment_engine:
  type: jev_bm25_then_select_multiple
  params:
    model: jev/jev-latest
    threshold: 0.4
    aggregate_over: 1000
    prompt: |
      Which categories best describe the query?
      {query}
    retrieval:
      fields: [title^9.4, description^4]  # Candidate-search BM25 fields/weights
```


### LLM hallucinate-then-resolve

Reference: https://softwaredoug.com/blog/2026/08/10/hypothetical-classifications

Hallucinate a set of hypothetical classifications using an LLM (ie OpenAI).
Then resolve those hallucinations to the most likely categories in the corpus, returning a list
of categories

- Embed every legal, non-empty category with `resolve_model` (for example,
  `all-MiniLM-L6-v2`). The full category vocabulary is indexed; it is not capped.
- Select one shared set of up to `num_samples` real category names from the
  corpus. Reject a sample when its cosine similarity to any selected sample is
  greater than or equal to `max_sample_sim`. Reuse this set in every query prompt.
- For each query, call the LLM once to generate a `list[str]` of hypothetical
  classifications. These may not match category values in the corpus.
- Resolve each hypothetical classification against the full category embedding
  index. Return unique categories whose cosine similarity is strictly greater
  than `similarity_threshold`.

```yaml
enrichment_engine:
  type: hallucinate_then_resolve
  params:
    model: openai/gpt-5.4-nano
    similarity_threshold: 0.9
    num_samples: 5
    max_sample_sim: 0.1
    resolve_model: all-MiniLM-L6-v2
    system_prompt: |
        Your task is to create novel, never seen before, furniture,
        home goods, or hardware classifications that best fit a search query.
    prompt: |
       Some inspiration on what these look like is at the bottom.

       Here is the user's request:

       {query}

       {category_name} classifications might look like:
       {samples}
```

Prompt caching is enabled by default. Set `params.cache: false` to bypass the
AutoEnricher prompt cache and the enricher's in-memory query cache. The
`uv run run --no-cache` option also disables caching for this engine.

Resolution works as follows:

```python
classification_embeddings = # embedded earlier with SentenceTransformer

def resolve(hallucinated: list[str],  # Hypothetical classifications from the model
            similarity_threshold):  # Threshold per above
    actuals = []
    for halluc_class in hallucinated:
        query_embedding = model.encode(halluc_class)
        dot_prods = np.dot(classifications_embeddings, query_embedding)
        actual = classifications_list[np.argmax(dot_prods)]   # Get the most similar classification
        while actual in actuals:       # If already in actuals, move down the list
            dot_prods[np.argmax(dot_prods)] = -1
            actual = classifications_list[np.argmax(dot_prods)]
        dot_prod = np.max(dot_prods)
        if dot_prod > similarity_threshold:   # Accept only if above threshold
            actuals.append(actual)
    return actuals
```

### Choice option behavior

#### Vocabulary-based choice engines when `choices` is omitted

For `llm_choice`, `jev_choice_single`, and `jev_choice_multiple`, omitting or
providing an empty `choices` mapping uses the most frequent category values in
the corpus vocabulary as options. The strategy caps that vocabulary at 300
values. `llm_choice` can use that full vocabulary; Jev receives at most 255
options. `jev_choice_single` reserves one option for `Unknown`, so it uses at
most 254 category values and maps `Unknown` to no category. `jev_choice_multiple`
has no synthetic `Unknown` option and excludes an explicit `Unknown` from its
results.

This vocabulary behavior does not apply to the BM25-then-select engines, whose
options come from each query's BM25 matches.

#### LLM choice

`choices` is a YAML dictionary mapping each choice name to its description. It should not be a block-scalar string.

- The choice names are passed to the model at the end of the prompt.
- The legal values are also used as `Literal` values in the structured output schema.
- If `pad_missing_choices` is true, corpus vocabulary values without descriptions
  remain legal options and appear without descriptions in the prompt.


#### Jev choice engines

- The prompt becomes Jev's Choice `instructions`.
- The option names and descriptions are passed in the `criteria` mapping.
- Unlabeled options use `None` descriptions, as in Jev's
  [Choice documentation](https://docs.typesafe.ai/primitives/choice).
