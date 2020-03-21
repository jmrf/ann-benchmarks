.phony clean

help:
	@echo "    mnist-faiss-flat"
	@echo "        Run MNIST (d=768) with FAISS-FLAT-IVF"




mnist-faiss-ivf:
	python run \
		--dataset fashion-mnist-784-euclidean \
		--definitions algos_faiss.yaml \
		--algorithm faiss-ivf \
		--cpu-number 4 \
		--local \



