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
β̂: [1.0512 0.2314 1.4429]
R²: [0.7896]
Adjusted R²: [0.7864]

--- Homoskedastic ---
SE: [0.0697 0.0706 0.0672]
p-values: [0.0000 0.0012 0.0000]

--- Robust (HC3) ---
SE: [0.0637 0.0755 0.0703]
p-values: [0.0000 0.0025 0.0000]
```

Development:

```
uvx pre-commit install
uvx pre-commit autoupdate
uvx ruff format .
uvx ruff check --fix .
uvx ty check

uv venv
source .venv/bin/activate
uv pip install .
```
