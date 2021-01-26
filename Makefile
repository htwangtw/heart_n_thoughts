# source env/bin/activate
install: requirements.txt
	test -d env || virtualenv -p python3 env; \
	. env/bin/activate; \
	pip install -r requirements.txt; \

# Regenerate PCA on control and patient groups separately
results/basic_pca/pca_control-vs-patients.* : data/derivatives/nback_derivatives/task-nbackmindwandering_probes.tsv bin/basic_pca.py
	. env/bin/activate; \
	python bin/basic_pca.py

clean :
	rm -rf results/*
