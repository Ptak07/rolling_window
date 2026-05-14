.PHONY: r-doc r-build r-test r-all py-build py-test py-all all

r-doc:
	Rscript -e "devtools::document()"

r-sync-headers:
	mkdir -p inst/include
	cp include/*.hpp inst/include/

r-build: r-sync-headers r-doc
	R CMD INSTALL .

r-test:
	Rscript -e "tinytest::test_package('robustrolling')"

r-all: r-build r-test

py-build:
	.venv/bin/pip install -e py_package/ --no-cache-dir

py-test:
	.venv/bin/pytest py_package/tests/ -v --tb=short

py-all: py-build py-test

all: r-all py-all
