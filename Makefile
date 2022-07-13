MAIN_FILES = $(basename $(wildcard *main.py))
TEST_FILES = $(basename $(wildcard *_test.cpp))

all: test run

run:
	python main.py

test:
	python test.py

no:
	for T in $(TEST_FILES); do \
	python $$T\
	; done

checkstyle:
	pylint *.py
