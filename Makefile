.PHONY: setup
setup:
	python3 -m venv .venv
	./.venv/bin/python3 -m ensurepip
	./.venv/bin/python3 -m pip install -r requirements.txt
