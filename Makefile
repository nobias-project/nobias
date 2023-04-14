black:
	python -m black .

gitall:
	git add .
	@read -p "Enter commit message: " message; 	git commit -m "$$message"
	git push

build:
	sphinx-build -a docs/source/ docs/build