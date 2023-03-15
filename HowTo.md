#### Some notes on how to deploy this
Followed this tutorial https://packaging.python.org/en/latest/tutorials/packaging-projects/
The pwd is on my google
Dont forget to update the version number in setup.py -- currently is 0.1.4
And to change the testpipy to pipy in the twine upload
And to skip existing versions with
python3 -m twine upload --skip-existing --repository pypi dist/*

Steps 
1. Change version in pyproject.toml
2. Build with
   1.  python3 -m build
3. deploy with
   1. python3 -m twine upload --skip-existing --repository pypi dist/*
``