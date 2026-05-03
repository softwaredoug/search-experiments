## Runner tests

This code is architected so that the command line arguments match 1-1 with a "Runner" object

Notice the runners.py and the exp/runners/ module

Notice the tests in tests/test_runners.py

### How runner tests should work

Runner tests are end to end. They use OPENAI and test fixtures when agentic strategies are involved. They run the entire pipeline, from strategy instantiation to evaluation.

They're meant to stimulate the code as if run from the command line.

### Small amounts of data. Small models.

- Use the smaller doug_blog dataset. Or tmdb if needed.
- Use num_queries=10 or something small to keep runtime down.
- If testing training, set rounds=1 or 2 (or pass --rounds) to keep runtime down.
- Use tiny models like gpt-5-mini at the most

Even with these, tests will be slow. That's ok

### Test with yml configs

Your job is to use test fixture yaml configs that define an experiment and its params. Use the fixtures in tests/fixtures/configs

### Ensure tests are integration tests

Do minimal or no mocking. Test on direct APIs (ie OPENAI, etc).

Expect OPENAI_API_KEY to be set. Tests should fail if not.
