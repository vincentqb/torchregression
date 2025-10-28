# torchregression

Regression with pytorch

```
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate
uv pip install .
```

Development

```
uv tool run pre-commit install
uv tool run pre-commit autoupdate
uv tool run ruff format .
uv tool run ruff check --fix .
uv tool run ty
```
