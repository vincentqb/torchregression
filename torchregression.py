import scipy.stats as st
import torch


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


def train_ols(model, X, y, *, lr=0.05, epochs=2000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()


def compute_se(model, X, y, *, robust):
    """
    Covariance & Standard Error computation
    robust: "none" | "HC0" | .. .| "HC5"
    """
    n, p = X.shape
    with torch.no_grad():
        # Prepare design matrix
        y_hat = model(X)
        resid = y - y_hat

        if model.linear.bias is None:
            X_design = X
            beta = model.linear.weight.flatten()
        else:
            X_design = torch.cat([X, torch.ones(n, 1)], dim=1)
            beta = torch.cat([model.linear.weight.flatten(), model.linear.bias])

        # (X'X)^(-1)
        XtX_inv = torch.inverse(X_design.T @ X_design)

        if robust == "none":
            # Homoskedastic OLS
            sigma2 = (resid.T @ resid) / (n - p)
            cov_beta = sigma2 * XtX_inv

        else:
            # Heteroskedasticity-consistent covariance
            # HC0: X' diag(e_i^2) X
            # HC1: scale by n/(n - p)
            # HC3: divide by (1 - h_ii)^2, where h = X(X'X)^(-1)X'
            Xh = X_design @ XtX_inv @ X_design.T
            h_ii = torch.diag(Xh)

            if robust == "HC0":
                scale = torch.ones_like(h_ii)
            elif robust == "HC1":
                scale = n / (n - p) * torch.ones_like(h_ii)
            elif robust == "HC2":
                scale = 1.0 / (1 - h_ii)
            elif robust == "HC3":
                scale = 1.0 / (1 - h_ii) ** 2
            elif robust == "HC4":
                h_mean = h_ii.mean()
                delta_i = torch.minimum(torch.tensor(4.0), h_ii / h_mean)
                scale = 1.0 / (1 - h_ii) ** delta_i
            elif robust == "HC5":
                h_mean = h_ii.mean()
                delta_i = torch.minimum(torch.tensor(4.0), (h_ii / h_mean) ** 0.7)
                scale = 1.0 / (1 - h_ii) ** delta_i
            else:
                raise ValueError("robust must be one of 'none', 'HC0'...'HC5'")

            # Compute X' diag(e_i^2 * scale_i) X
            w = (resid.squeeze() ** 2 * scale).view(-1, 1)
            X_weighted = X_design * torch.sqrt(w)
            cov_beta = XtX_inv @ (X_weighted.T @ X_weighted) @ XtX_inv

        se = torch.sqrt(torch.diag(cov_beta))
        return beta, cov_beta, se


def regression_summary(model, X, y, *, robust="none"):
    # Compute regression summary
    n, p = X.shape
    with torch.no_grad():
        y_hat = model(X)
        resid = y - y_hat
        ssr = torch.sum((y_hat - y.mean()) ** 2)
        sst = torch.sum((y - y.mean()) ** 2)
        r2 = ssr / sst
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    beta, cov, se = compute_se(model, X, y, robust=robust)
    t_stat = beta / se
    p_values = torch.from_numpy(2 * (1 - st.t.cdf(torch.abs(t_stat).detach().numpy(), df=n - p)))
    return {
        "beta": beta,
        "se": se,
        "t_stat": t_stat,
        "p_values": p_values,
        "cov": cov,
        "r2": r2.reshape(1),
        "adj_r2": adj_r2.reshape(1),
        "residuals": resid.squeeze(),
    }


if __name__ == "__main__":
    import numpy as np

    X, y, beta_true = make_data(n=200, p=3)

    model = LinearRegression(X.shape[1], bias=False)
    train_ols(model, X, y)
    stats_homosked = regression_summary(model, X, y, robust="none")

    robust = "HC3"
    stats = regression_summary(model, X, y, robust=robust)

    with np.printoptions(precision=4, floatmode="fixed", suppress=False):
        print("--- Ground Truth ---")
        print("β̂:", beta_true.flatten().detach().numpy())

        print("\n--- OLS ---")
        print("β̂:", stats_homosked["beta"].detach().numpy())
        print("R²:", stats_homosked["r2"].detach().numpy())
        print("Adjusted R²:", stats_homosked["adj_r2"].detach().numpy())

        print("\n--- Homoskedastic ---")
        print("SE:", stats_homosked["se"].detach().numpy())
        print("p-values:", stats_homosked["p_values"].detach().numpy())

        print(f"\n--- Robust ({robust}) ---")
        print("SE:", stats["se"].detach().numpy())
        print("p-values:", stats["p_values"].detach().numpy())
