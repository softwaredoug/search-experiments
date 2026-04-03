.PHONY: profile

profile:
	uv run python -m cProfile -o profile.prof -m prf.runner --strategy prf --num-queries 50 --seed 42
	uv run snakeviz profile.prof
