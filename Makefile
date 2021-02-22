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
features : data/derivatives/nback_derivatives/task-nbackmindwandering_probes.tsv bin/make_features.py
	. env/bin/activate; \
	python bin/make_features.py

test:
	. env/bin/activate; \
	pip install -e .
	pytest

clean :
	rm -rf results/*
