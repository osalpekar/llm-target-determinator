install:
	pip install -r requirements.txt

lint:
	usort format .
	black --preview --line-length 79 .
	flake8 .
