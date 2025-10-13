.PHONY: install test run
install:
	pip install -r requirements.txt

test:
	pytest -q

run:
	docker-compose up --build
