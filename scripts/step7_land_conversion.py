"""EarthLens — Step 7: Spatial samples and persistent land conversion.

Usage (VS Code terminal, with BOTH original CSVs next to this script):
    py -m pip install pandas numpy matplotlib
    py EarthLens_Step7_Spatial_Land_Conversion.py

Outputs are written to EarthLens_Step7_Output. The spatial plots show
SAMPLED POINTS, not continuous satellite rasters, areas, or hotspots.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -------- 1) Find the two original CSV files (do not edit originals) --------
ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [ROOT / 'data']


def locate_csv(fragment):
    matches = []
    for folder in SEARCH_DIRS:
        if folder.exists():
            matches.extend(folder.glob(f'*{fragment}*.csv'))
    matches = list(dict.fromkeys(p.resolve() for p in matches))
    if not matches:
        raise FileNotFoundError(
            f'Could not find CSV containing {fragment!r}. '
            f'Put both original EarthLens CSV files in the data/ folder.'
        )
    # Prefer the original dataset filename over derived outputs.
    matches.sort(key=lambda p: (len(p.name), p.name))
    return matches[0]


urban_file = locate_csv('Urban_Growth_Analysis')
complete_file = locate_csv('Complete_Dataset')
urban = pd.read_csv(urban_file)
complete = pd.read_csv(complete_file)

required_urban = {'Record_Type', 'Baseline_Year', 'Confirmation_Year',
                  'Source_Class', 'Converted_To', 'Persistent_Conversion_km2'}
required_complete = {'task', 'split', 'year', 'label', 'converted',
                     'Baseline_Year', 'Target_Year', 'longitude', 'latitude'}
for name, df, required in [('Urban Growth', urban, required_urban),
                           ('Complete Dataset', complete, required_complete)]:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{name} is missing expected columns: {sorted(missing)}')

OUT = ROOT / 'outputs' / 'step7'
OUT.mkdir(parents=True, exist_ok=True)

print(f'Urban Growth file: {urban_file.name} ({len(urban):,} rows)')
print(f'Complete Dataset: {complete_file.name} ({len(complete):,} rows)')

# -------- 2) Persistent conversion: aggregate existing AREA records --------
# The 56 records are class-by-period summaries, NOT 56 individual pixels.
conv = urban[urban['Record_Type'].eq('persistent_conversion')].copy()
conv['Persistent_Conversion_km2'] = pd.to_numeric(
    conv['Persistent_Conversion_km2'], errors='coerce')
conv = conv[
    (conv['Persistent_Conversion_km2'] >= 0) &
    (conv['Baseline_Year'] >= 0) &
    (conv['Confirmation_Year'] >= 0)
].copy()
conv['Baseline_Year'] = conv['Baseline_Year'].astype(int)
conv['Confirmation_Year'] = conv['Confirmation_Year'].astype(int)
conv['Period'] = conv['Baseline_Year'].astype(str) + '→' + conv['Confirmation_Year'].astype(str)

period_summary = (
    conv.groupby(['Baseline_Year', 'Confirmation_Year', 'Period'], as_index=False)
    ['Persistent_Conversion_km2'].sum()
    .sort_values('Baseline_Year')
)
source_summary = (
    conv.groupby('Source_Class', as_index=False)['Persistent_Conversion_km2'].sum()
    .sort_values('Persistent_Conversion_km2', ascending=False)
)
total_period_area = source_summary['Persistent_Conversion_km2'].sum()
source_summary['Percent_of_summed_periods'] = np.where(
    total_period_area > 0,
    100 * source_summary['Persistent_Conversion_km2'] / total_period_area,
    0.0,
)
period_summary.to_csv(OUT / 'EarthLens_Conversion_By_Period.csv', index=False)
source_summary.to_csv(OUT / 'EarthLens_Conversion_By_Source.csv', index=False)

# -------- 3) Geographic sample points — NOT area or true density --------
class_samples = complete[
    complete['task'].eq('classification') & (complete['year'] == 2025)
].copy()

# 2023 baseline -> 2025 target: 300 positive and 300 negative samples.
conversion_samples = complete[
    complete['task'].eq('susceptibility') &
    (complete['Baseline_Year'] == 2023) &
    (complete['Target_Year'] == 2025) &
    complete['converted'].isin([0, 1])
].copy()

for name, subset in [('2025 classification', class_samples),
                     ('2023–2025 conversion', conversion_samples)]:
    if subset.empty:
        raise ValueError(f'No rows for {name}; check the uploaded CSV version.')
    if not subset[['longitude', 'latitude']].apply(np.isfinite).all().all():
        raise ValueError(f'Non-finite coordinates found in {name}.')
    if (subset['longitude'].abs() > 180).any() or (subset['latitude'].abs() > 90).any():
        raise ValueError(f'Invalid latitude/longitude in {name}.')

class_samples[['year', 'label', 'longitude', 'latitude']].to_csv(
    OUT / 'EarthLens_2025_Classification_Samples.csv', index=False)
conversion_samples[['Baseline_Year', 'Target_Year', 'converted',
                    'longitude', 'latitude']].to_csv(
    OUT / 'EarthLens_Spatial_Samples_2023_2025.csv', index=False)

# -------- 4) Draw land conversion bars --------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), layout='constrained')
axes[0].bar(period_summary['Period'], period_summary['Persistent_Conversion_km2'],
            color='#167D8D')
axes[0].set_title('Persistent land conversion to built-up, by 2-year period')
axes[0].set_xlabel('Baseline → confirmation year')
axes[0].set_ylabel('Reported conversion area (km²)')
axes[0].tick_params(axis='x', labelrotation=45)
axes[0].grid(axis='y', alpha=.2)
for i, v in enumerate(period_summary['Persistent_Conversion_km2']):
    axes[0].text(i, v + .05, f'{v:.2f}', ha='center', fontsize=8)

nonzero = source_summary.query('Persistent_Conversion_km2 > 0').copy()
axes[1].barh(nonzero['Source_Class'].iloc[::-1],
             nonzero['Persistent_Conversion_km2'].iloc[::-1], color='#D88B45')
axes[1].set_title('Source land-cover classes → built-up (summed periods)')
axes[1].set_xlabel('Reported conversion area summed over periods (km²)')
axes[1].grid(axis='x', alpha=.2)
for i, v in enumerate(nonzero['Persistent_Conversion_km2'].iloc[::-1]):
    axes[1].text(v + .2, i, f'{v:.4f}', va='center', fontsize=9)
axes[1].set_xlim(0, max(nonzero['Persistent_Conversion_km2']) * 1.23)
fig.suptitle('EarthLens | Land conversion evidence from annual summaries',
             fontsize=14, fontweight='bold')
fig.savefig(OUT / 'EarthLens_Land_Conversion_Charts.png', dpi=220)
plt.close(fig)

# -------- 5) Draw point locations (not a continuous spatial change map) --------
label_names = {0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded vegetation',
               4: 'Crops', 5: 'Shrub & scrub', 6: 'Built', 7: 'Bare',
               8: 'Snow & ice'}
class_samples['land_cover'] = class_samples['label'].map(label_names).fillna('Other')
colors = {'Water': '#277DA1', 'Grass': '#76B85A', 'Crops': '#D7A642',
          'Built': '#BA4D45', 'Bare': '#7E7770', 'Trees': '#17624D',
          'Flooded vegetation': '#6497A5', 'Shrub & scrub': '#8F9A49',
          'Snow & ice': '#B4CCDF', 'Other': '#222222'}
fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8), layout='constrained')
for cat in sorted(class_samples['land_cover'].unique()):
    pts = class_samples[class_samples['land_cover'].eq(cat)]
    axes[0].scatter(pts['longitude'], pts['latitude'], s=10, alpha=.62,
                    label=f'{cat} (n={len(pts)})', color=colors.get(cat, '#444444'),
                    linewidths=0)
axes[0].set_title(f'2025 classification samples (n={len(class_samples):,})')
axes[0].legend(loc='best', fontsize=8, markerscale=1.7, framealpha=.9)

for converted, label, col in [(0, 'Not converted', '#4285B4'),
                               (1, 'Converted to built', '#C54F50')]:
    pts = conversion_samples[conversion_samples['converted'].eq(converted)]
    axes[1].scatter(pts['longitude'], pts['latitude'], s=12, alpha=.65,
                    label=f'{label} (n={len(pts)})', color=col, linewidths=0)
axes[1].set_title(f'2023 → 2025 balanced conversion samples (n={len(conversion_samples):,})')
axes[1].legend(loc='best', fontsize=9, markerscale=1.5)

for ax in axes:
    ax.set_xlabel('Longitude (degrees E)')
    ax.set_ylabel('Latitude (degrees N)')
    ax.grid(alpha=.16)
    avg_lat = class_samples['latitude'].mean()
    ax.set_aspect(1 / np.cos(np.deg2rad(avg_lat)), adjustable='box')
    ax.set_xlim(31.35, 31.98)
    ax.set_ylim(29.86, 30.13)
fig.suptitle('EarthLens | Georeferenced sample locations — NOT a full land-cover map',
             fontsize=14, fontweight='bold')
fig.savefig(OUT / 'EarthLens_Spatial_Sample_Maps.png', dpi=220)
plt.close(fig)

# -------- 6) Print an interpretable summary and save it --------
positive = int((conversion_samples['converted'] == 1).sum())
negative = int((conversion_samples['converted'] == 0).sum())
bare = float(source_summary.loc[source_summary['Source_Class'].eq('bare'),
                                'Persistent_Conversion_km2'].sum())
crops = float(source_summary.loc[source_summary['Source_Class'].eq('crops'),
                                 'Persistent_Conversion_km2'].sum())
summary = f'''EARTHLENS — STEP 7 SUMMARY
=========================
Urban Growth rows used: {len(conv)} persistent conversion summaries
Complete Dataset rows: {len(complete):,}
2025 classification samples: {len(class_samples):,}
2023→2025 conversion samples: {len(conversion_samples):,}
  Converted=1: {positive} samples
  Converted=0: {negative} samples

Sums across the available 2-year windows (NOT unique area):
  Bare → Built:  {bare:.6f} km² ({bare/total_period_area*100:.4f}%)
  Crops → Built: {crops:.6f} km² ({crops/total_period_area*100:.4f}%)
  Total across source classes/periods: {total_period_area:.6f} km²

INTERPRETATION LIMITS
1. The 2-year windows overlap. Never interpret their summed areas as unique
   land converted in one non-overlapping period or equate to annual net growth.
2. Sample points were selected for tasks, so their count cannot estimate
   regional land-cover percentages, conversion prevalence, or hotspots.
3. A point scatter plot is NOT a continuous satellite change map. A valid
   pixel-level area/change map would require raster imagery and an ROI polygon.
4. 'Bare' is a satellite class, not proof of ground ownership, vacant lots,
   ecological quality, or exact construction location.
5. 'Converted=1' denotes the provided task label; this script does not
   independently validate conversion against satellite ground truth.
'''
(OUT / 'EarthLens_Step7_Interpretation.txt').write_text(summary, encoding='utf-8')
print('\n' + summary)
print(f'\nSaved outputs under: {OUT}')
for p in sorted(OUT.iterdir()):
    print(' -', p.name)