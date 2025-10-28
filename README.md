# torchregression

Regression with pytorch:

```
curl -LsSf https://astral.sh/uv/install.sh | sh

uv run python torchregression.py
```

Sample output:

```
--- Ground Truth ---
β̂: [1.0104 0.1100 1.4311]

--- OLS ---
β̂: [1.0513 0.2308 1.4415 0.0105]
R²: [0.7886]
Adjusted R²: [0.7854]

--- Homoskedastic ---
SE: [0.0697 0.0707 0.0679 0.0707]

--- Robust (HC3) ---
SE: [0.0643 0.0773 0.0720 0.0732]
```

Development:

```
uv tool run pre-commit install
uv tool run pre-commit autoupdate
uv tool run ruff format .
uv tool run ruff check --fix .
uv tool run ty

uv venv
source .venv/bin/activate
uv pip install .
```
