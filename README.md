### Packages installation
Install virtualenv if you don't have it
```
sudo apt-get install virtualenv
```

Create python 3.6 virtual environment and activate it
```
virtualenv -p python3.6 venv
source venv/bin/activate
```
Install dependencies
```
pip install -r requirements.txt
```

### Run tests
```
cd tests
python -m pytest tests.py -v
```