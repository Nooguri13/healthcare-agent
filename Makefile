.PHONY: install index demo eval mcp test clean

install:
	pip install -r requirements.txt

index:
	python -m src.ingest

demo:
	python scripts/demo.py

eval:
	python -m evals.run_eval

mcp:
	python -m mcp_server.server

test:
	pytest -q

clean:
	rm -rf .index __pycache__ */__pycache__ .pytest_cache
