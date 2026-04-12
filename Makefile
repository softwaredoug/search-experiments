.PHONY: profile test

profile:
	uv run python -m cProfile -o profile.prof -m prf.runner --strategy prf --num-queries 10 --seed 42 --workers 1
	uv run snakeviz profile.prof

test:
	scripts/test_bm25_consistency.sh --strategy bm25
	scripts/test_bm25_consistency.sh --strategy bm25_doubleidf
	scripts/test_bm25_consistency.sh --strategy bm25_reweighed
