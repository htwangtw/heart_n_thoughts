# source env/bin/activate
install: env requirements.txt
	test -d env || virtualenv -p python3 env; \
	. env/bin/activate; \
	pip install -r requirements.txt; \

# Regenerate PCA on control and patient groups separately
results/pca_control-vs-patients.png : data/task-nbackmindwandering_probes.tsv bin/pca.py
	. env/bin/activate; \
	python bin/pca.py