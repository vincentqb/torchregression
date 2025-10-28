import scipy.stats as st
import torch
from rich.box import HORIZONTALS
from rich.console import Console
from rich.progress import track
from rich.table import Table


def make_data(n=200, p=3, noise=1.0, seed=0):
    # Simple data generator for testing
    torch.manual_seed(seed)
    X = torch.randn(n, p)
    beta_true = torch.randn(p, 1)
    y = X @ beta_true + noise * torch.randn(n, 1)
    return X, y, beta_true


class LinearRegression(torch.nn.Module):
    def __init__(self, n_features, *, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1, bias=bias)

    def forward(self, x):
        return self.linear(x)


def train_ols(model, X, y, *, lr=0.05, epochs=10, batch_size=32):
    """Train OLS using mini-batches"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    n = len(X)
    for _ in track(range(epochs), description="Training..."):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X[idx], y[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()


def compute_se(model, X, y, *, robust="none", batch_size=256):
    """
    Covariance & Standard Error computation (batched)
    robust: "none" | "HC0" | "HC1" | "HC2" | "HC3"
    """
    n, p = X.shape
    device = X.device

    with torch.no_grad():
        # Storage for accumulated matrices
        add_bias = model.linear.bias is not None
        p_eff = p + (1 if add_bias else 0)

        XtX = torch.zeros((p_eff, p_eff), device=device)
        rss = 0.0
        n_obs = 0

        # First pass: accumulate XtX and RSS
        for i in range(0, n, batch_size):
            xb = X[i : i + batch_size]
            yb = y[i : i + batch_size]
            nb = len(xb)

            y_hat = model(xb)
            resid = yb - y_hat
            n_obs += nb
            rss += (resid.T @ resid).item()

            if add_bias:
                Xb = torch.cat([xb, torch.ones(nb, 1, device=device)], dim=1)
            else:
                Xb = xb

            XtX += Xb.T @ Xb

        XtX_inv = torch.inverse(XtX)

        if robust == "none":
            sigma2 = rss / (n_obs - p_eff)
            cov_beta = sigma2 * XtX_inv

        else:
            # Second pass for heteroskedasticity-consistent covariance
            cov_beta = torch.zeros_like(XtX)
            for i in range(0, n, batch_size):
                xb = X[i : i + batch_size]
                yb = y[i : i + batch_size]
                nb = len(xb)

                y_hat = model(xb)
                resid = yb - y_hat

                if add_bias:
                    Xb = torch.cat([xb, torch.ones(nb, 1, device=device)], dim=1)
                else:
                    Xb = xb

                Xh = Xb @ XtX_inv @ Xb.T
                h_ii = torch.diag(Xh)

                if robust == "HC0":
                    scale = torch.ones_like(h_ii)
                elif robust == "HC1":
                    scale = n / (n - p_eff) * torch.ones_like(h_ii)
                elif robust == "HC2":
                    scale = 1.0 / (1 - h_ii)
                elif robust == "HC3":
                    scale = 1.0 / (1 - h_ii) ** 2
                else:
                    raise ValueError(f"Unsupported robust type {robust}")

                w = (resid.squeeze() ** 2 * scale).view(-1, 1)
                X_weighted = Xb * torch.sqrt(w)
                cov_beta += X_weighted.T @ X_weighted

            cov_beta = XtX_inv @ cov_beta @ XtX_inv

        # Extract coefficients and SE
        if add_bias:
            beta = torch.cat([model.linear.weight.flatten(), model.linear.bias])
        else:
            beta = model.linear.weight.flatten()

        se = torch.sqrt(torch.diag(cov_beta))

        # For completeness, collect all residuals
        resid_full = []
        for i in range(0, n, batch_size):
            xb = X[i : i + batch_size]
            yb = y[i : i + batch_size]
            resid_full.append(yb - model(xb))
        resid_full = torch.cat(resid_full)

        return beta, cov_beta, se, resid_full


def regression_summary(model, X, y, *, robust="none", batch_size=None):
    """
    Regression summary supporting batched evaluation.
    X: [B, n, p] or [n, p]
    y: [B, n, 1] or [n, 1]
    """
    # Ensure batch dimension
    if X.dim() == 2:
        X = X.unsqueeze(0)
        y = y.unsqueeze(0)

    B, n, p = X.shape

    # Storage for results
    betas, covs, ses, t_stats, p_vals, r2s, adj_r2s, residuals_list = [], [], [], [], [], [], [], []

    for b in range(B):
        Xb, yb = X[b], y[b]

        beta, cov, se, resid = compute_se(model, Xb, yb, robust=robust, batch_size=batch_size)
        y_hat = model(Xb)
        ssr = torch.sum((y_hat - yb.mean()) ** 2)
        sst = torch.sum((yb - yb.mean()) ** 2)
        r2 = ssr / sst
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        t_stat = beta / se
        p_value = torch.from_numpy(2 * (1 - st.t.cdf(torch.abs(t_stat).detach().numpy(), df=n - p)))

        # Collect
        betas.append(beta)
        covs.append(cov)
        ses.append(se)
        t_stats.append(t_stat)
        p_vals.append(p_value)
        r2s.append(r2)
        adj_r2s.append(adj_r2)
        residuals_list.append(resid)

    # Stack results
    return {
        "beta": torch.stack(betas),
        "se": torch.stack(ses),
        "t_stat": torch.stack(t_stats),
        "p_values": torch.stack(p_vals),
        "cov": torch.stack(covs),
        "r2": torch.stack(r2s),
        "adj_r2": torch.stack(adj_r2s),
        "residuals": torch.stack(residuals_list),
    }


def pretty_summary(d):
    table = Table(box=HORIZONTALS)
    table.add_column("Key", justify="right")
    table.add_column("Values", justify="center")

    for key, values in d.items():
        values = values.flatten().detach().numpy().tolist()
        table.add_row(key, "   ".join(f"{value:+1.4f}" for value in values))

    console = Console()
    console.print(table)


if __name__ == "__main__":
    X, y, beta_true = make_data(n=200, p=3)

    model = LinearRegression(X.shape[1], bias=False)
    train_ols(model, X, y, batch_size=32, epochs=50)

    stats_homosked = regression_summary(model, X, y, robust="none", batch_size=64)
    robust = "HC3"
    stats = regression_summary(model, X, y, robust=robust, batch_size=64)

    pretty_summary(
        {
            "Ground Truth β": beta_true,
            "Estimated β̂": stats_homosked["beta"],
            "Homoskedastic σ̂": stats_homosked["se"],
            "Homoskedastic p": stats_homosked["p_values"],
            # "Homoskedastic t statistics": stats_homosked["t_stat"],
            f"Robust ({robust}) σ̂": stats["se"],
            f"Robust ({robust}) p": stats["p_values"],
            # f"Robust ({robust}) t statistics": stats_homosked["t_stat"],
            "R²": stats_homosked["r2"],
            "Adjusted R²": stats_homosked["adj_r2"],
        }
    )
