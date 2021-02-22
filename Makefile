# source env/bin/activate
install: requirements.txt
	test -d env || virtualenv -p python3 env; \
	. env/bin/activate; \
	pip install -r requirements.txt; \

# report data property
datareport :
	. env/bin/activate; \
	python bin/data_property.py

# Regenerate PCA on control and patient groups separately
results/basic_pca/pca_control-vs-patients.* : data/derivatives/nback_derivatives/task-nbackmindwandering_probes.tsv bin/basic_pca.py
	. env/bin/activate; \
	python bin/basic_pca.py

results/task_performance.tsv : data/derivatives/nback_derivatives/task-nbackmindwandering_performance.tsv bin/parse_taskperform.py
	. env/bin/activate; \
	python bin/parse_taskperform.py

results/task-nback_summary.tsv : bin/create_summary.py results/basic_pca/pca_control-vs-patients.tsv results/task_performance.tsv
	. env/bin/activate; \
	python bin/create_summary.py

clean :
	rm -rf results/*
