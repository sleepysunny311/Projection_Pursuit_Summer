# Projection Pursuit Task 1

## Setup

1. Create a virtual environment

```
# using venv
python -m venv .venv
source .venv/bin/activate   # on Windows, run the corresponding script

# alternatively, using conda
conda create -n hw2env python=3.9 pip
conda activate hw2env
```

2. Install poetry

```
pip install poetry
```

3. Install dependencies

```
poetry install
```

## Running the tests for MP and OMP

```
python testing.py
```

All the tests parameteres are defined in the 'Configs' folder. Just change the parameters in the `testing.py` file to run the tests for different parameters.

You can also generate different testings by changing the parameters in `Configs_Generator.ipynb` and running it.



