## Runner tests

This code is architected so that the command line arguments match 1-1 with a "Runner" object

Notice the runners.py and the exp/runners/ module

Notice the tests in tests/test_runners.py

### How runner tests should work

Runner tests are end to end. They use OPENAI and test fixtures when agentic strategies are involved. They run the entire pipeline, from strategy instantiation to evaluation.

They're meant to stimulate the code as if run from the command line.
