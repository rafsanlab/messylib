# messylib
General codes (mostly python) usable for your projects. Codes are organised into each individual folders or modules according to their theme, each module will have their own spesific requirements.txt. So install the dependencies according to your module usage, this to keep your environment light.
```
messylib
 L messylib
    L module1
      L requirements.txt
      L codes.py
      ...
    L module2
      L requirements.txt
      L codes.py
      ...
    ...
```
## Hot to use
1. Clone this repo:
```
git clone https://github.com/rafsanlab/messylib.git
```
2. Install requirements based on which modules to use (optional):
```
pip install -r messylib/visualisation/requirements.txt
```
3. Start using it in your project:
```
from messylib.visualisation.dataframe import plot_dataframe
```

## Disclaimer
Test and validate the code using your own data before using the results for your production or any other case. I do not bear responsibilities of any incorrect outputs generated from any codes from this library. Treat this as a personal repo that you can edit. I recommend you to fork this repo and edited accordingly for your own use case.
