.PHONY: profile

profile:
	uv run python -m cProfile -o profile.prof -m prf.runner --strategy prf --num-queries 10 --seed 42 --workers 1
	uv run snakeviz profile.prof
