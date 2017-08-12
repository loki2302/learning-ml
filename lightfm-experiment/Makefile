SHELL := /bin/bash

init:
	virtualenv env; \
	source env/bin/activate; \
	pip install -r requirements.txt; \

deinit:
	source env/bin/activate; \
	deactivate; \
	rm -rf ./env; \

run:
	source env/bin/activate; \
	python app.py

requirements:
	source env/bin/activate; \
	pip freeze >requirements1.txt; \
	grep -v "pkg-resource" requirements1.txt > requirements.txt
	rm requirements1.txt
