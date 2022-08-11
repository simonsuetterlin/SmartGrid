MAIN_FILES = $(basename $(wildcard *main.py))

all: test run

run:
	python3 main.py

