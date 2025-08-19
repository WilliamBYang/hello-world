import numpy as np
import pandas as pd

try:
    import cvxpy as cp  # Optional; we provide a fallback if not available
    HAS_CVXPY = True
except Exception:  # pragma: no cover
    HAS_CVXPY = False


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip a Series to its [lower, upper] quantiles."""
    if series.empty:
        return series
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)


def zscore_series(series: pd.Series) -> pd.Series:
    """Standard z-score with population std. Returns zeros if std is non-finite."""
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def robust_zscore(series: pd.Series) -> pd.Series:
    """Winsorize then z-score."""
    return zscore_series(winsorize_series(series, 0.01, 0.99))


def ewma(series: pd.Series, half_life: int) -> float:
    if len(series) == 0:
        return 0.0
    lam = np.exp(np.log(0.5) / max(half_life, 1))
    weights = np.array([lam ** (len(series) - 1 - i) for i in range(len(series))], dtype=float)
    weights /= weights.sum()
    return float(np.dot(series.values, weights))


def one_hot_encode(values: pd.Series) -> pd.DataFrame:
    return pd.get_dummies(values.astype(str))


# ------------------------------------------------------------
# Factor Construction (Barra + Characteristic factors: skew, kurt, VaR, ES)
# ------------------------------------------------------------
class BarraFactors:
    def __init__(self, panel: pd.DataFrame):
        """
        Expected panel schema (long format):
        - date: pd.Timestamp or str
        - ticker: str
        - close: float
        - returns: float (simple return)
        - mcap: float (market capitalization)
        - book_equity: float
        - roe: float
        - turnover: float (share/value turnover)
        - industry: str
        """
        df = panel.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(['ticker', 'date'], inplace=True)
        self.df = df

    # ---------- Core rolling stats ----------
    def _rolling_volatility(self, window: int = 60) -> pd.Series:
        return self.df.groupby('ticker')['returns'].transform(lambda x: x.rolling(window, min_periods=max(20, window // 3)).std())

    def _rolling_skewness(self, window: int = 60) -> pd.Series:
        # pandas.rolling.skew implements an unbiased estimator
        return self.df.groupby('ticker')['returns'].transform(lambda x: x.rolling(window, min_periods=max(20, window // 3)).skew())

    def _rolling_kurtosis_excess(self, window: int = 60) -> pd.Series:
        # pandas.rolling.kurt returns Fisher's definition (excess kurtosis)
        return self.df.groupby('ticker')['returns'].transform(lambda x: x.rolling(window, min_periods=max(20, window // 3)).kurt())

    def _rolling_var_es(self, window: int = 60, alpha: float = 0.95) -> tuple[pd.Series, pd.Series]:
        # Historical VaR/ES at tail 1-alpha, using simple rolling window
        def var_es(arr: np.ndarray) -> tuple[float, float]:
            if len(arr) == 0 or np.all(np.isnan(arr)):
                return np.nan, np.nan
            valid = arr[~np.isnan(arr)]
            if len(valid) < max(20, window // 3):
                return np.nan, np.nan
            q = np.quantile(valid, 1.0 - alpha)
            var95 = -float(q)  # positive loss convention
            tail = valid[valid <= q]
            if len(tail) == 0:
                es95 = np.nan
            else:
                es95 = -float(tail.mean())
            return var95, es95

        def rolling_apply(series: pd.Series) -> pd.DataFrame:
            vals = series.rolling(window, min_periods=max(20, window // 3)).apply(lambda x: var_es(np.asarray(x, dtype=float))[0], raw=False)
            es_vals = series.rolling(window, min_periods=max(20, window // 3)).apply(lambda x: var_es(np.asarray(x, dtype=float))[1], raw=False)
            return pd.DataFrame({'VaR': vals, 'ES': es_vals})

        out = self.df.groupby('ticker')['returns'].apply(rolling_apply)
        # The above produces a multiindex with ticker/date; align back
        out = out.reset_index(level=0, drop=True)
        var_series = out['VaR']
        es_series = out['ES']
        return var_series, es_series

    def _rolling_liquidity_adv20(self) -> pd.Series:
        self.df['dollar_vol'] = self.df['mcap'] * self.df['turnover']
        return self.df.groupby('ticker')['dollar_vol'].transform(lambda x: x.rolling(20, min_periods=5).mean())

    def _momentum_12m_1m(self) -> pd.Series:
        # 12-1 momentum: last 252 days excluding last 21 trading days
        def calc_mom(r: pd.Series) -> pd.Series:
            r1 = (1 + r).rolling(252, min_periods=126).apply(lambda x: float(np.prod(x)), raw=True)
            r2 = (1 + r).rolling(21, min_periods=10).apply(lambda x: float(np.prod(x)), raw=True)
            mom = r1 / r2
            mom = np.log(mom.replace(0, np.nan))
            return mom

        return self.df.groupby('ticker')['returns'].transform(calc_mom)

    # ---------- Public API ----------
    def build_daily_exposures(self, window: int = 60, alpha: float = 0.95) -> dict:
        """
        Build per-date matrices for cross-sectional regression and optimization.

        Returns a dict[date] -> {
            'X': exposures matrix (n, k_style + k_ind),
            'X_style': style factor exposures (n, k_style),
            'X_ind': industry dummies (n, k_ind),
            'ret_next': next-day returns vector (n,),
            'tickers': list[str],
            'style_cols': list[str],
            'ind_cols': list[str],
        }
        """
        df = self.df.copy()

        # Classical Barra-style factors
        df['size'] = -np.log(df['mcap'].replace({0: np.nan}))
        df['value'] = np.log((df['book_equity'] / df['mcap']).replace({0: np.nan}))
        df['quality'] = df['roe']
        df['volatility'] = self._rolling_volatility(window=window)
        df['liquidity'] = np.log(self._rolling_liquidity_adv20().replace({0: np.nan}))
        df['momentum'] = self._momentum_12m_1m()

        # Characteristic factors
        df['skewness'] = self._rolling_skewness(window=window)
        df['kurtosis'] = self._rolling_kurtosis_excess(window=window)
        var_s, es_s = self._rolling_var_es(window=window, alpha=alpha)
        df['var_95'] = var_s
        df['es_95'] = es_s

        # Next-day realized returns as target
        df['ret_next'] = df.groupby('ticker')['returns'].shift(-1)

        style_cols = [
            'size', 'value', 'momentum', 'quality', 'volatility', 'liquidity',
            'skewness', 'kurtosis', 'var_95', 'es_95'
        ]

        dates = sorted(df['date'].dropna().unique())
        date_to_data: dict = {}

        for current_date in dates:
            sub = df[df['date'] == current_date].copy()
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=['ret_next', 'industry'])

            # Require a minimum cross-section
            if len(sub) < 80:
                continue

            # Winsorize + z-score each style exposure
            for col in style_cols:
                sub[col] = robust_zscore(sub[col])

            # Industry dummies
            X_ind = one_hot_encode(sub['industry'])

            # Neutralize style exposures vs. industries (cross-sectional) via ridge
            X_style_neutralized = []
            Z = X_ind.values
            ridge = 1e-4 * np.eye(Z.shape[1]) if Z.shape[1] > 0 else None
            for col in style_cols:
                y = sub[col].values
                if Z.shape[1] == 0:
                    resid = y.copy()
                else:
                    beta = np.linalg.pinv(Z.T @ Z + ridge) @ (Z.T @ y)
                    resid = y - Z @ beta
                X_style_neutralized.append(resid)
            X_style_neutralized = np.column_stack(X_style_neutralized)

            X_full = np.column_stack([X_style_neutralized, Z])

            date_to_data[current_date] = {
                'X': X_full,
                'X_style': X_style_neutralized,
                'X_ind': Z,
                'ret_next': sub['ret_next'].values,
                'tickers': list(sub['ticker'].values),
                'style_cols': style_cols,
                'ind_cols': list(X_ind.columns),
            }

        return date_to_data


# ------------------------------------------------------------
# Factor Return Estimation (Cross-Sectional Ridge)
# ------------------------------------------------------------
def estimate_daily_factor_returns(date_to_data: dict) -> pd.DataFrame:
    rows = []
    for the_date, obj in date_to_data.items():
        X = obj['X']
        y = obj['ret_next']
        lam = 1e-4
        beta = np.linalg.pinv(X.T @ X + lam * np.eye(X.shape[1])) @ (X.T @ y)

        row = {'date': the_date}
        for i, name in enumerate(obj['style_cols']):
            row[name] = beta[i]
        for j, ind_name in enumerate(obj['ind_cols']):
            row[f"IND_{ind_name}"] = beta[len(obj['style_cols']) + j]
        rows.append(row)

    df_beta = pd.DataFrame(rows).set_index('date').sort_index()
    return df_beta


# ------------------------------------------------------------
# Risk Model and Forecasts
# ------------------------------------------------------------
def build_risk_model_and_forecast(df_beta: pd.DataFrame, half_life: int = 60) -> tuple[list[str], np.ndarray, np.ndarray]:
    style_cols = [c for c in df_beta.columns if not c.startswith('IND_')]

    # Expected factor returns via EWMA
    mu_style = np.array([ewma(df_beta[c].dropna(), half_life=half_life) for c in style_cols])

    # Factor covariance with light shrinkage
    style_ret_mat = df_beta[style_cols].dropna()
    if len(style_ret_mat) < 10:
        F_style = np.eye(len(style_cols)) * 1e-4
    else:
        F_style = np.cov(style_ret_mat.values.T)
        alpha = 0.2
        F_style = alpha * F_style + (1 - alpha) * np.diag(np.diag(F_style))

    return style_cols, mu_style, F_style


def estimate_specific_risk(date_to_data: dict, df_beta: pd.DataFrame) -> dict:
    specific_var_by_date: dict = {}
    for the_date, obj in date_to_data.items():
        X = obj['X']
        y = obj['ret_next']

        beta_row = df_beta.loc[the_date]
        style_cols = obj['style_cols']
        ind_cols = [f"IND_{c}" for c in obj['ind_cols']]
        beta_vec = np.array(list(beta_row[style_cols].values) + list(beta_row[ind_cols].reindex(ind_cols, fill_value=0.0).values))

        y_hat = X @ beta_vec
        resid = y - y_hat
        var_est = np.maximum(resid.var(ddof=min(len(resid) - 1, 1)), 1e-6)
        specific_var_by_date[the_date] = np.full(len(y), var_est)

    return specific_var_by_date


# ------------------------------------------------------------
# Optimization (with cvxpy or a lightweight fallback)
# ------------------------------------------------------------
def _optimize_with_cvxpy(obj: dict, mu_style: np.ndarray, F_style: np.ndarray,
                         gamma: float, leverage: float, max_weight: float) -> tuple[np.ndarray, np.ndarray]:
    X_style = obj['X_style']
    X_ind = obj['X_ind']
    n, _ = X_style.shape

    alpha_vec = X_style @ mu_style

    Sigma_factor = X_style @ F_style @ X_style.T
    diag_specific = np.maximum(np.diag(Sigma_factor).mean() * 0.3, 1e-6)
    Sigma = Sigma_factor + np.eye(n) * diag_specific

    w = cp.Variable(n)
    objective = cp.Maximize(alpha_vec @ w - gamma * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 0.0, cp.norm1(w) <= leverage, w <= max_weight, w >= -max_weight]
    if X_ind.shape[1] > 0:
        constraints.append(X_ind.T @ w == 0.0)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False, max_iter=20000)
    except Exception:
        pass
    if w.value is None:
        try:
            prob.solve(solver=cp.SCS, verbose=False, max_iters=8000)
        except Exception:
            pass
    if w.value is None:
        raise RuntimeError("cvxpy optimization did not converge")
    return np.asarray(w.value).ravel(), alpha_vec


def _project_to_constraints(raw_w: np.ndarray, X_ind: np.ndarray, leverage: float, max_weight: float) -> np.ndarray:
    """Simple projection: enforce industry neutrality and dollar neutrality via least-squares,
    then clip to box and scale to leverage (L1). Iterate a few times."""
    w = raw_w.copy()

    def project_linear_equalities(w_vec: np.ndarray) -> np.ndarray:
        A_list = [np.ones_like(w_vec)]
        if X_ind.size > 0:
            A_list.append(X_ind.T)  # shape (k_ind, n)
        A = np.vstack(A_list)
        # Solve min ||w' - w||^2 s.t. A w' = 0
        # Closed-form: w' = w - A^T (A A^T)^+ (A w)
        Aw = A @ w_vec
        G = A @ A.T
        try:
            G_pinv = np.linalg.pinv(G)
        except Exception:
            G_pinv = np.eye(G.shape[0])
        correction = A.T @ (G_pinv @ Aw)
        return w_vec - correction

    for _ in range(5):
        w = project_linear_equalities(w)
        w = np.clip(w, -max_weight, max_weight)
        l1 = np.sum(np.abs(w))
        if l1 > 1e-12:
            w *= (leverage / l1)
        # Re-enforce equalities after scaling/clipping
        w = project_linear_equalities(w)

    return w


def _optimize_fallback(obj: dict, mu_style: np.ndarray, F_style: np.ndarray,
                       gamma: float, leverage: float, max_weight: float) -> tuple[np.ndarray, np.ndarray]:
    X_style = obj['X_style']
    X_ind = obj['X_ind']
    alpha_vec = X_style @ mu_style
    # Start from alpha-ranked weights
    if alpha_vec.std(ddof=0) == 0 or np.isnan(alpha_vec.std(ddof=0)):
        raw_w = np.zeros_like(alpha_vec)
    else:
        raw_w = (alpha_vec - alpha_vec.mean()) / (alpha_vec.std(ddof=0) + 1e-12)
    # Scale target intensity using a crude risk proxy to avoid extreme bets
    n = len(raw_w)
    scale = min(1.0, np.sqrt(n) / (np.linalg.norm(raw_w) + 1e-9))
    raw_w *= scale
    w = _project_to_constraints(raw_w, X_ind, leverage=leverage, max_weight=max_weight)
    return w, alpha_vec


def optimize_portfolio(obj: dict, mu_style: np.ndarray, F_style: np.ndarray,
                       gamma: float = 10.0, leverage: float = 1.0, max_weight: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    if HAS_CVXPY:
        try:
            return _optimize_with_cvxpy(obj, mu_style, F_style, gamma, leverage, max_weight)
        except Exception:
            # Fallback to projection-based heuristic
            return _optimize_fallback(obj, mu_style, F_style, gamma, leverage, max_weight)
    else:
        return _optimize_fallback(obj, mu_style, F_style, gamma, leverage, max_weight)


# ------------------------------------------------------------
# End-to-end Pipeline
# ------------------------------------------------------------
class BarraExtendedStrategy:
    def __init__(self, panel: pd.DataFrame, window: int = 60, alpha: float = 0.95):
        self.factor_builder = BarraFactors(panel)
        self.date_to_data = self.factor_builder.build_daily_exposures(window=window, alpha=alpha)
        if len(self.date_to_data) == 0:
            raise ValueError("Insufficient data to build exposures. Check inputs and window length.")
        self.df_beta = estimate_daily_factor_returns(self.date_to_data)
        self.style_cols, self.mu_style, self.F_style = build_risk_model_and_forecast(self.df_beta)
        self.specific_risk = estimate_specific_risk(self.date_to_data, self.df_beta)

    def rebalance_on(self, the_date: pd.Timestamp) -> pd.DataFrame:
        if the_date not in self.date_to_data:
            raise ValueError(f"No data for date {the_date}")
        obj = self.date_to_data[the_date]
        weights, alpha_vec = optimize_portfolio(
            obj,
            self.mu_style,
            self.F_style,
            gamma=10.0,
            leverage=1.0,
            max_weight=0.02,
        )
        res = pd.DataFrame({
            'ticker': obj['tickers'],
            'weight': weights,
            'alpha': alpha_vec,
        })
        res.sort_values('weight', ascending=False, inplace=True)
        return res


# ------------------------------------------------------------
# Demo with synthetic data (replace with real panel for production)
# ------------------------------------------------------------
def make_synthetic_panel(num_stocks: int = 300, num_days: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2019-01-01', periods=num_days)
    tickers = [f"S{str(i).zfill(4)}" for i in range(num_stocks)]
    industries = [f"IND{j}" for j in range(12)]
    ticker_to_industry = {t: rng.choice(industries) for t in tickers}

    rows = []
    for t in tickers:
        price = 100.0 * np.exp(rng.normal(0, 0.05))
        mcap = rng.lognormal(mean=10, sigma=1.0)
        book = mcap * rng.uniform(0.3, 1.1)
        roe = rng.normal(0.12, 0.06)
        turnover = rng.lognormal(-2.0, 0.6)

        for d in dates:
            ret = rng.normal(0.0003, 0.02)
            price *= (1 + ret)
            mcap *= (1 + ret)
            turnover = max(1e-6, turnover * np.exp(rng.normal(0, 0.05)))
            roe = np.clip(roe + rng.normal(0, 0.004), -0.6, 0.6)
            book = max(1e3, book * np.exp(rng.normal(0, 0.01)))

            rows.append({
                'date': d,
                'ticker': t,
                'close': price,
                'returns': ret,
                'mcap': mcap,
                'book_equity': book,
                'roe': roe,
                'turnover': turnover,
                'industry': ticker_to_industry[t],
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Demo: build and run a rebalance on the last date
    panel = make_synthetic_panel()
    strategy = BarraExtendedStrategy(panel, window=60, alpha=0.95)
    last_date = sorted(strategy.date_to_data.keys())[-1]
    weights_df = strategy.rebalance_on(last_date)

    print(f"Rebalance on {pd.to_datetime(last_date).date()} (top longs):")
    print(weights_df.head(10)[['ticker', 'weight', 'alpha']])
    print("Top shorts:")
    print(weights_df.tail(10)[['ticker', 'weight', 'alpha']])

