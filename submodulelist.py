"""import the submodule_requirements - a dict contains the paths to requirements.txt file"""

import os

currentdir = os.getcwd()
moduledir = os.path.join(currentdir, "messylib")

submodules = [
    i for i in os.listdir(moduledir) if not i.startswith(".") and not i.startswith("_")
]
print("Available submodules:", submodules)

submodule_requirements = {}
for submodule in submodules:
    submodule_path = os.path.join(moduledir, submodule)
    if "requirements.txt" in os.listdir(submodule_path):
        submodule_requirements[submodule] = os.path.join(
            submodule_path, "requirements.txt"
        )
