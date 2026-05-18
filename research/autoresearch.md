# Autoresearch in ranking results

## Overfitting to training data

Without any validation guardrails, we see issues overfitting to training data, as expected.

```
uv run run --strategy configs/codegen/codegen_no_guards.yml --dataset wands -rounds 10
```

[Run here](/runs/codegen/wands/codegen_no_guards/20260501_155216)

We plateau quickly when run on full query.

![no guards](/assets/No_validation_full_data.png)


We can see overfitting when we look at the non-test queries.

![no guards](/assets/No_validation_test_data.png)

## BM25 run

```
uv run train --strategy configs/codegen/codegen_minimarco.yml --dataset minimarco 
```

On Minimarco, we see a steady climb

![no guards](/assets/minimarco_lexical_retrieval.png)

On MSMarco, we see a plateau

![no guards](/assets/msmarco_lexical_retrieval.png)
