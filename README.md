# torchregression

Regression with pytorch:

```
curl -LsSf https://astral.sh/uv/install.sh | sh

uv run python torchregression.py
```

Sample output:

```
Training... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
 ─────────────────────────────────────────────────────
                    Key             Values
 ─────────────────────────────────────────────────────
         Ground Truth β   +1.0104   +0.1100   +1.4311
            Estimated β̂   +1.0251   +0.2020   +1.4594
     Homoskedastic SE σ̂   +0.0698   +0.0706   +0.0673
  Homoskedastic p value   +0.0000   +0.0047   +0.0000
     Robust (HC3) SE σ̂:   +0.0652   +0.0746   +0.0705
   Robust (HC3) p value   +0.0000   +0.0073   +0.0000
                     R²             +0.7856
            Adjusted R²             +0.7823
 ─────────────────────────────────────────────────────
```

Development:

```
uvx pre-commit install
uvx pre-commit autoupdate
uvx ruff format .
uvx ruff check --fix .
uvx ty check
uvx codespell

uv venv
source .venv/bin/activate
uv pip install .
```
