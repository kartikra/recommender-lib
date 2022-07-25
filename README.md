# recommender-lib

Build and evalauted standard recommendation algorithms


Inspired by -
- [Microsoft Recommenders](https://github.com/microsoft/recommenders)
- [Frank Kane Sundog](https://sundog-education.com/recsys/)


## Installation instructions

Install poetry by following instructions from [here](https://python-poetry.org/docs/#installation)

After poetry is installed do the following -
``` sh
git clone git@github.com:kartikra/recommender-lib.git && cd recommender-lib
poetry install 
```


## Setting up jupyter notebook
Before running [jupyter notebooks](/notebooks/01-recommender-evaluation.ipynb) in this project make sure that poetry virtual environment is installed as its own jupyter kernel
``` sh
poetry shell
python3 -m ipykernel install --name "Recsys" --user
```
After that do `jupyter notebook` and select the kernel "Recsys" for running notebook

## Training Jobs
- [basic_evaluation](recommender_lib/jobs/basic_evaluation.py)
