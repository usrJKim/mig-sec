#!/bin/bash

python3 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install tqdm matplotlib pandas
deactivate
