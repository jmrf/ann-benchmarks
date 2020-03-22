.PHONY: clean

JOBS ?= 2

help:
	@echo "	formatter"
	@echo "		Apply black formatting to code."
	@echo "	lint"
	@echo "		Lint code with flake8, and check if black formatter should be applied."
	@echo "	types"
	@echo "		Check for type errors using pytype."
	@echo "	pyupgrade"
	@echo "		Uses pyupgrade to upgrade python syntax."
	@echo "	mnist-faiss-flat"
	@echo "		Run MNIST (d=768) with FAISS-FLAT-IVF"


mnist-faiss-ivf:
	python run.py \
		--dataset fashion-mnist-784-euclidean \
		--definitions algos_faiss.yaml \
		--algorithm faiss-ivf \
		--cpu-number 4 \
		--local \

mnist-faiss:
	python run.py \
		--dataset fashion-mnist-784-euclidean \
		--definitions algos_faiss.yaml \
		--cpu-number 4 \
		--local \

formatter:
	black . --exclude tests/

lint:
	flake8 . --exclude tests/
	black --check . --exclude tests/

types:
	# https://google.github.io/pytype/
	pytype --keep-going ann_benchmarks --exclude ann_benchmarks/tests

pyupgrade:
	find .  -name '*.py' | grep -v 'proto\|eggs\|docs' | xargs pyupgrade --py36-plus

test: clean
	# OMP_NUM_THREADS can improve overral performance using one thread by process (on tensorflow), avoiding overload
	OMP_NUM_THREADS=1 pytest tests -n $(JOBS) --cov ann_benchmarks


clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	find . -name '__pycache__' | xargs rm -r
	rm -rf build/
	rm -rf .pytype/
	rm -rf dist/
	rm -rf docs/_build
	# rm -rf *egg-info
	# rm -rf pip-wheel-metadata
