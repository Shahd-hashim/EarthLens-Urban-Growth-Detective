"""EarthLens Step 5: tests and visualization (standalone local script)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, theilslopes
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox

# Locate original Urban Growth CSV, no changes to original files.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
folder = ROOT / "outputs" / "step5"
folder.mkdir(parents=True, exist_ok=True)
matches = list(DATA.glob('*Urban_Growth_Analysis*.csv'))
if not matches:
    raise FileNotFoundError('Place the EarthLens_Urban_Growth_Analysis CSV in the data/ folder.')
df = pd.read_csv(matches[0])
metric = 'Confident_Built_km2'

annual_test = df.loc[df['Record_Type'] == 'annual_growth', ['Year', metric]].copy()
annual_test = annual_test.dropna().sort_values('Year')
annual_test['Year'] = annual_test['Year'].astype(int)
ax = annual_test['Year'].to_numpy()
ay = annual_test[metric].to_numpy()
if len(ax) < 4 or len(ax) != len(np.unique(ax)):
    raise ValueError('Annual data must have at least four distinct years.')

# 1. Annual exact Kendall trend test and robust Theil-Sen magnitude
mk = kendalltau(ax, ay, method='exact')
sen = theilslopes(ay, ax, alpha=0.95)

# 2. Monthly seasonality and residual diagnostic
monthly_test = df.loc[df['Record_Type'] == 'monthly_growth', ['Date', metric]].copy()
monthly_test['Date'] = pd.to_datetime(monthly_test['Date'])
monthly_test = monthly_test[(monthly_test['Date'] >= '2016-01-01') &
                            (monthly_test['Date'] < '2026-01-01')].sort_values('Date')
ms = monthly_test.set_index('Date')[metric].asfreq('MS')
if len(ms) != 120 or ms.isna().any():
    raise ValueError('Expected 120 non-missing monthly observations, 2016–2025.')
stl = STL(ms, period=12, robust=True).fit()
lb_p12 = acorr_ljungbox(stl.resid, lags=[12], return_df=True)['lb_pvalue'].iloc[0]

# 3. Season-aware trend statistic. Each year is one complete 12-month block.
monthly_test['Year'] = monthly_test['Date'].dt.year
monthly_test['Month'] = monthly_test['Date'].dt.month
wide = monthly_test.pivot(index='Year', columns='Month', values=metric)
wide = wide.reindex(index=range(2016, 2026), columns=range(1,13))
if wide.isna().any().any():
    raise ValueError('Every calendar month of 2016–2025 is required.')
values = wide.to_numpy()
n_years = len(values)

def within_month_sign_stat(v):
    return int(sum(np.sign(v[j, :] - v[i, :]).sum()
                   for i in range(len(v) - 1) for j in range(i + 1, len(v))))

observed_stat = within_month_sign_stat(values)
# Year-block permutations preserve within-year relationships between months.
rng = np.random.default_rng(111)
B = 20000
extreme = 0
for _ in range(B):
    perm_stat = within_month_sign_stat(values[rng.permutation(n_years), :])
    extreme += abs(perm_stat) >= abs(observed_stat)
monthly_perm_p = (extreme + 1) / (B + 1)

# 4. A descriptive seasonal Theil-Sen slope from comparisons in matching months.
seasonal_slopes = [
    (values[j, col] - values[i, col]) / (j - i)
    for col in range(12)
    for i in range(n_years - 1)
    for j in range(i + 1, n_years)
]
monthly_sen = float(np.median(seasonal_slopes))

print('=== EARTHLENS STEP 5 ===')
print(f'Annual Kendall tau: {mk.statistic:.4f}')
print(f'Annual exact Kendall p-value: {mk.pvalue:.9f}')
print(f'Annual Sen slope: {sen.slope:.3f} km2/year')
print(f'Annual 95% slope CI: [{sen.low_slope:.3f}, {sen.high_slope:.3f}] km2/year')
print(f'Monthly season-aware sign statistic: {observed_stat}')
print(f'Monthly year-block permutation p-value (B={B}): {monthly_perm_p:.7f}')
print(f'Monthly within-month median slope: {monthly_sen:.3f} km2/year')
print(f'STL residual Ljung-Box p-value (12 lags): {lb_p12:.4f}')

# 5. A clear, presentation-friendly plot showing the yearly trend slope.
fig, chart = plt.subplots(figsize=(10, 5))
chart.plot(ax, ay, marker='o', label='Annual built-up estimates', color='royalblue')
chart.plot(ax, sen.intercept + sen.slope * ax, '--',
           label=f'Theil-Sen trend (+{sen.slope:.2f} km²/year)', color='darkorange')
chart.set_title('EarthLens — Urban Trend and Statistical Evidence')
chart.set_xlabel('Year')
chart.set_ylabel('Estimated Built-up Area (km²)')
chart.grid(alpha=0.3)
chart.legend()
fig.tight_layout()
fig.savefig(folder / 'EarthLens_Statistical_Trend.png', dpi=250)
plt.close(fig)

pd.DataFrame([
    ['Annual exact Kendall', 'tau', mk.statistic],
    ['Annual exact Kendall', 'p_value', mk.pvalue],
    ['Annual Theil-Sen', 'slope_km2_per_year', sen.slope],
    ['Annual Theil-Sen', '95pct_CI_low', sen.low_slope],
    ['Annual Theil-Sen', '95pct_CI_high', sen.high_slope],
    ['Monthly year-block permutation', 'p_value', monthly_perm_p],
    ['Monthly within-month slopes', 'median_slope_km2_per_year', monthly_sen],
    ['STL residual Ljung-Box', 'p_value_at_12_lags', lb_p12],
], columns=['Analysis', 'Metric', 'Value']).to_csv(folder / 'EarthLens_Statistical_Results.csv', index=False)