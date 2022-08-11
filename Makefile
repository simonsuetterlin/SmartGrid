MAIN_FILES = $(basename $(wildcard *main.py))

all: run

run:
	python3 main.py

