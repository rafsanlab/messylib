# messylib
General codes (mostly python) usable for your projects. Codes are organised into each individual folders or modules according to their theme, each module will have their own spesific requirements.txt. So install the dependencies according to your module usage.
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
* alternatively, you can check what modules are available by importing a dictionary name `submodule_requirements` from `submodulelist.py` that will also contain the path to each corresponding requirements.txt.

3. Start using it in your project:
```
from messylib.visualisation.dataframes import plot_df_cols
```

## Disclaimer
Test and validate the code with your own data before using the results or the outputs from this repository for your production or any other case. I bear no responsibilities of any incorrect outputs generated from any codes from this library. Treat this as a personal repo that you can edit. I recommend you to fork this repo and edited accordingly for your own use case.
