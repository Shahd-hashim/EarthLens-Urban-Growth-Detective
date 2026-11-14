"""EarthLens | Step 6: Satellite observation quality & metric sensitivity.

Standalone local script. Reads from ../data/EarthLens_Urban_Growth_Analysis*.csv,
then run: py EarthLens_Step6_Data_Quality.py
No Streamlit; input files are never modified.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, theilslopes

# These are EXPLORATORY screening thresholds, not independently validated quality standards.
MIN_COVERAGE_PCT = 95.0
MIN_IMAGE_COUNT = 4
START = '2016-01-01'
END_EXCLUSIVE = '2026-01-01'

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
folder = DATA
candidates = sorted(DATA.glob('EarthLens_Urban_Growth_Analysis*.csv'))
if not candidates:
    candidates = sorted(folder.glob('*Urban_Growth_Analysis*.csv'))
if not candidates:
    raise FileNotFoundError('Place your EarthLens_Urban_Growth_Analysis CSV in the data/ folder.')
source = candidates[0]
output = ROOT / 'outputs' / 'step6'
output.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source)
required = ['Record_Type', 'Date', 'Year', 'Satellite_Images', 'Valid_Observation_km2',
            'Confident_Built_km2', 'Modal_Built_km2']
missing = set(required) - set(df.columns)
if missing:
    raise ValueError('Required columns missing: ' + ', '.join(sorted(missing)))

# Clean only the fields/record types used, treating -999 as an unavailable marker.
monthly = df.loc[df['Record_Type'].eq('monthly_growth'),
                 ['Date', 'Satellite_Images', 'Valid_Observation_km2', 'Confident_Built_km2']].copy()
monthly['Date'] = pd.to_datetime(monthly['Date'], errors='raise')
monthly = monthly.loc[(monthly.Date >= START) & (monthly.Date < END_EXCLUSIVE)].sort_values('Date')
for c in ['Satellite_Images', 'Valid_Observation_km2', 'Confident_Built_km2']:
    monthly[c] = pd.to_numeric(monthly[c], errors='coerce').replace(-999, np.nan)
if len(monthly) != 120 or monthly.Date.duplicated().any() or monthly.isna().any().any():
    raise ValueError('Expected 120 complete, unique monthly records during 2016-2025.')
if (monthly[['Satellite_Images', 'Valid_Observation_km2']] <= 0).any().any():
    raise ValueError('Expected positive observation area and satellite image counts.')

annual = df.loc[df.Record_Type.eq('annual_growth'),
                ['Year', 'Valid_Observation_km2', 'Confident_Built_km2', 'Modal_Built_km2']].copy()
annual = annual.replace(-999, np.nan)
annual = annual.sort_values('Year')
annual['Year'] = annual['Year'].astype(int)
if len(annual) != 9 or annual.isna().any().any():
    raise ValueError('Expected nine complete annual observations for 2017-2025.')

# The denominator is the annual valid footprint (~1067.25 km2),
# treated as a consistent reference, not a proof of pixel-level quality.
reference_area = float(annual['Valid_Observation_km2'].median())
monthly['Coverage_Percent'] = 100 * monthly['Valid_Observation_km2'] / reference_area
if (monthly['Coverage_Percent'] > 100.01).any():
    raise ValueError('Coverage exceeds reference region: check area definitions.')
monthly['Low_Coverage'] = monthly['Coverage_Percent'] < MIN_COVERAGE_PCT
monthly['Few_Images'] = monthly['Satellite_Images'] < MIN_IMAGE_COUNT
monthly['QC_Pass'] = ~(monthly['Low_Coverage'] | monthly['Few_Images'])
monthly['QC_Reason'] = np.select(
    [monthly['Low_Coverage'] & monthly['Few_Images'], monthly['Low_Coverage'], monthly['Few_Images']],
    ['Low coverage + few images', 'Low coverage', 'Few images'], default='Pass'
)
monthly['Year'] = monthly.Date.dt.year
monthly['Month'] = monthly.Date.dt.month
monthly.to_csv(output / 'EarthLens_Monthly_Quality_Check.csv', index=False)

# Check whether the season-matched slope remains similar after screening.
def seasonal_median_slope(data):
    slopes = []
    for month, group in data.groupby('Month'):
        x = group['Year'].to_numpy()
        y = group['Confident_Built_km2'].to_numpy()
        for i in range(len(group)-1):
            for j in range(i+1, len(group)):
                if x[j] != x[i]:
                    slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return float(np.median(slopes)) if slopes else np.nan

monthly_slope_all = seasonal_median_slope(monthly)
monthly_slope_qc = seasonal_median_slope(monthly.loc[monthly.QC_Pass])

# Annual sensitivity analysis: different definitions of built-up area.
rows = []
for name in ['Confident_Built_km2', 'Modal_Built_km2']:
    x = annual.Year.to_numpy()
    y = annual[name].astype(float).to_numpy()
    slope = theilslopes(y, x, alpha=0.95)
    mk = kendalltau(x, y, method='exact')
    rows.append({
        'Metric': name,
        '2017_km2': y[0], '2025_km2': y[-1],
        'Net_Change_km2': y[-1] - y[0],
        'Theil_Sen_Slope_km2_per_year': slope.slope,
        'Slope_95CI_Low': slope.low_slope, 'Slope_95CI_High': slope.high_slope,
        'Kendall_Tau': mk.statistic, 'Kendall_p_value': mk.pvalue
    })
sensitivity = pd.DataFrame(rows)
sensitivity.to_csv(output / 'EarthLens_Annual_Metric_Sensitivity.csv', index=False)

summary = pd.DataFrame([
    {'Metric': 'Monthly observations', 'Value': len(monthly)},
    {'Metric': 'Reference region km2', 'Value': reference_area},
    {'Metric': 'Months under 95% coverage', 'Value': int(monthly.Low_Coverage.sum())},
    {'Metric': 'Months with under 4 images', 'Value': int(monthly.Few_Images.sum())},
    {'Metric': 'Months failing either screen', 'Value': int((~monthly.QC_Pass).sum())},
    {'Metric': 'Minimum observed coverage percent', 'Value': monthly.Coverage_Percent.min()},
    {'Metric': 'Season-matched monthly slope all (km2/year)', 'Value': monthly_slope_all},
    {'Metric': 'Season-matched monthly slope screened (km2/year)', 'Value': monthly_slope_qc},
])
summary.to_csv(output / 'EarthLens_Quality_Summary.csv', index=False)

# Figure A: monthly spatial observation coverage and provisional threshold.
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(monthly.Date, monthly.Coverage_Percent, color='steelblue', marker='.', linewidth=1.2,
        label='Valid observation coverage (%)')
flagged = monthly.loc[~monthly.QC_Pass]
ax.scatter(flagged.Date, flagged.Coverage_Percent, color='crimson', zorder=4,
           label='Flagged month (coverage or image count)')
ax.axhline(MIN_COVERAGE_PCT, linestyle='--', color='darkorange',
           label=f'Illustrative coverage screen ({MIN_COVERAGE_PCT:.0f}%)')
ax.set(title='EarthLens - Monthly Observation Coverage Quality', xlabel='Date',
       ylabel='Valid observation coverage (%)', ylim=(0, 105))
ax.legend(loc='lower left', fontsize=9)
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(output / 'EarthLens_Observation_Coverage.png', dpi=250)
plt.close(fig)

# Figure B: same years; two built-up definitions (not two independent datasets).
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(annual.Year, annual.Confident_Built_km2, '-o', color='royalblue',
        label='Confident built-up estimate')
ax.plot(annual.Year, annual.Modal_Built_km2, '-s', color='darkorange',
        label='Modal built-up estimate')
ax.set(title='EarthLens - Annual Metric Sensitivity (2017-2025)',
       xlabel='Year', ylabel='Estimated built-up area (km²)')
ax.grid(alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(output / 'EarthLens_Annual_Metric_Comparison.png', dpi=250)
plt.close(fig)

print('\n=== EARTHLENS STEP 6: DATA QUALITY & SENSITIVITY ===')
print('Input:', source.name)
print(f'Months analyzed: {len(monthly)} (2016-2025)')
print(f'Reference area: {reference_area:.2f} km2')
print(f'Months below {MIN_COVERAGE_PCT:.0f}% coverage: {monthly.Low_Coverage.sum()}')
print(f'Months with fewer than {MIN_IMAGE_COUNT} images: {monthly.Few_Images.sum()}')
print(f'Months flagged by either screen: {(~monthly.QC_Pass).sum()}')
print('Flagged months:')
print(flagged[['Date','Satellite_Images','Coverage_Percent','QC_Reason']].to_string(index=False,
      formatters={'Date': lambda d: d.strftime('%Y-%m'), 'Coverage_Percent': '{:.2f}'.format}))
print(f'Season-matched slope BEFORE QC: {monthly_slope_all:.3f} km2/year')
print(f'Season-matched slope AFTER QC:  {monthly_slope_qc:.3f} km2/year')
print('\nAnnual metrics:')
print(sensitivity[['Metric','Theil_Sen_Slope_km2_per_year','Kendall_p_value']].to_string(index=False))
print('\nResults folder:', output)
print('Outputs: 3 CSV files + 2 PNG charts; no original files changed.')