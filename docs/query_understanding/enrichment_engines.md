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

