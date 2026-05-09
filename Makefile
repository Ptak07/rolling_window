.PHONY: r-doc r-build r-all

r-doc:
	Rscript -e "devtools::document()"

r-test: 
	Rscript -e "tinytest::test_package('robustrolling')"

r-sync-headers:
	cp include/*.hpp inst/include/

r-build: r-sync-headers r-doc
	R CMD INSTALL .

r-all: r-build r-test
