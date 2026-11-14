"""EarthLens Step 8 | NASA Black Marble nighttime lights (2017--2025).

This script does NOT contain NASA observations. When run with an authenticated
Google Earth Engine account, it retrieves annual summaries of NASA VNP46A2
inside an approximate study region inferred from classification sample sites.
For an exact area comparison, supply the original region GeoJSON via --roi.

Run from VS Code, with the ORIGINAL EarthLens CSVs in the same directory:
    py -m pip install earthengine-api pandas numpy scipy matplotlib
    py EarthLens_Step8_NASA_BlackMarble.py --check
    py EarthLens_Step8_NASA_BlackMarble.py --project YOUR_EARTH_ENGINE_PROJECT_ID

Or with original project boundary (recommended):
    py EarthLens_Step8_NASA_BlackMarble.py --project YOUR_ID --roi original_region.geojson

Google Earth Engine is an external service requiring authorized access; this
script has been validated locally for syntax and input geometry only. It has
not been run against live NASA imagery in this environment.

NASA dataset: NASA/VIIRS/002/VNP46A2
Band: DNB_BRDF_Corrected_NTL (nighttime radiance, nW/cm2/sr)
Quality: Mandatory_Quality_Flag == 0 or 1 (as documented for EE collection).
Annual summary: median daily radiance per pixel, then regional mean across
valid pixels. Night lights are NOT direct measurements of built-up area.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "step8"
OUT.mkdir(parents=True, exist_ok=True)


def find_original_csv(fragment: str) -> Path:
    candidates = []
    for folder in (ROOT / "data",):
        if folder.is_dir():
            candidates.extend(folder.glob(f"*{fragment}*.csv"))
    candidates = list(dict.fromkeys(p.resolve() for p in candidates))
    if not candidates:
        raise FileNotFoundError(f"Original CSV containing '{fragment}' not found; place it in the data/ folder.")
    candidates.sort(key=lambda p: (len(p.name), p.name))
    return candidates[0]


def get_geometry_object(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("type") == "FeatureCollection":
        if len(raw.get("features", [])) != 1:
            raise ValueError("GeoJSON FeatureCollection must contain exactly one study-region polygon.")
        raw = raw["features"][0]["geometry"]
    elif raw.get("type") == "Feature":
        raw = raw["geometry"]
    if raw.get("type") not in ("Polygon", "MultiPolygon"):
        raise ValueError("The region must be a Polygon or MultiPolygon GeoJSON, not points or lines.")
    return raw


def approx_polygon_from_samples(complete: pd.DataFrame) -> dict:
    """Convex hull of 2025 classification sample coordinates, NOT source ROI."""
    needed = {"task", "year", "longitude", "latitude"}
    if missing := needed - set(complete.columns):
        raise ValueError(f"Missing columns in Complete Dataset: {sorted(missing)}")
    samples = complete[
        complete["task"].eq("classification") & (complete["year"] == 2025)
    ][["longitude", "latitude"]].dropna().drop_duplicates()
    coords = samples.to_numpy(dtype=float)
    if len(coords) < 3 or not np.isfinite(coords).all():
        raise ValueError("Insufficient finite georeferenced classification samples for ROI approximation.")
    hull = ConvexHull(coords)
    ring = coords[hull.vertices].round(7).tolist()
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def calculate_nasa_years(geometry: dict, project_id: str) -> pd.DataFrame:
    try:
        import ee
    except ImportError as exc:
        raise RuntimeError("Earth Engine library missing. Install: py -m pip install earthengine-api") from exc

    try:
        ee.Initialize(project=project_id)
    except Exception as first_error:
        print("Earth Engine isn't ready. Attempting one-time Google login...")
        try:
            ee.Authenticate(auth_mode="localhost")
            ee.Initialize(project=project_id)
        except Exception as exc:
            raise RuntimeError(
                "Earth Engine login/initialization failed. Register or enable your "
                "Earth Engine Cloud project and replace YOUR_ID with its exact project ID. "
                "Docs: https://developers.google.com/earth-engine/guides/auth"
            ) from exc

    roi = ee.Geometry(geometry)
    all_lights = (ee.ImageCollection("NASA/VIIRS/002/VNP46A2")
                  .filterBounds(roi))

    def select_quality(image):
        quality = image.select("Mandatory_Quality_Flag").lte(1)
        return (image.select("DNB_BRDF_Corrected_NTL")
                .updateMask(quality)
                .rename("NTL"))

    reducer = ee.Reducer.mean().combine(reducer2=ee.Reducer.count(), sharedInputs=True)
    year_features = []
    for year in range(2017, 2026):
        year_images = all_lights.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        annual_median = year_images.map(select_quality).median()
        stats = annual_median.reduceRegion(
            reducer=reducer,
            geometry=roi,
            scale=500,
            maxPixels=100_000_000,
            tileScale=4,
        )
        year_features.append(ee.Feature(None, {
            "Year": year,
            "NASA_NTL_Mean": stats.get("NTL_mean"),
            "NASA_Valid_Pixels": stats.get("NTL_count"),
            "NASA_Daily_Images": year_images.size(),
        }))

    print("Requesting NASA VNP46A2 annual summaries from Earth Engine...")
    features = ee.FeatureCollection(year_features).getInfo().get("features", [])
    rows = [f.get("properties", {}) for f in features]
    nasa = pd.DataFrame(rows)
    if nasa.empty:
        raise RuntimeError("Earth Engine returned no annual NASA results.")
    for col in ("Year", "NASA_NTL_Mean", "NASA_Valid_Pixels", "NASA_Daily_Images"):
        nasa[col] = pd.to_numeric(nasa[col], errors="coerce")
    nasa = nasa.sort_values("Year").reset_index(drop=True)
    if nasa["NASA_NTL_Mean"].isna().any() or (nasa["NASA_Valid_Pixels"] <= 0).any():
        raise RuntimeError("Some years have no valid NASA pixels; review the ROI and NASA quality mask.")
    return nasa


def create_comparison(nasa: pd.DataFrame, urban: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    needed = {"Record_Type", "Year", "Confident_Built_km2"}
    if missing := needed - set(urban.columns):
        raise ValueError(f"Missing Urban Growth columns: {sorted(missing)}")
    annual = urban[urban["Record_Type"].eq("annual_growth")].copy()
    annual["Year"] = pd.to_numeric(annual["Year"], errors="coerce")
    annual["Confident_Built_km2"] = pd.to_numeric(annual["Confident_Built_km2"], errors="coerce")
    annual = annual[annual["Year"].between(2017, 2025)].copy()
    merged = pd.merge(
        annual[["Year", "Confident_Built_km2"]],
        nasa[["Year", "NASA_NTL_Mean", "NASA_Valid_Pixels", "NASA_Daily_Images"]],
        on="Year", how="inner", validate="one_to_one"
    ).sort_values("Year")
    if len(merged) != 9:
        raise RuntimeError(f"Expected 9 aligned years, got {len(merged)}")

    # Equal 2017 index makes shapes comparable but units remain fundamentally different.
    first_built = merged["Confident_Built_km2"].iloc[0]
    first_ntl = merged["NASA_NTL_Mean"].iloc[0]
    merged["Urban_Index_2017_100"] = 100 * merged["Confident_Built_km2"] / first_built
    if first_ntl > 0:
        merged["NASA_Index_2017_100"] = 100 * merged["NASA_NTL_Mean"] / first_ntl
    else:
        merged["NASA_Index_2017_100"] = np.nan
    merged.to_csv(OUT / "EarthLens_NASA_vs_Urban_Annual.csv", index=False)
    nasa.to_csv(OUT / "EarthLens_NASA_Annual_Nighttime_Lights.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(merged["Year"], merged["Urban_Index_2017_100"], "-o", lw=2.4,
            label="Urban built-up estimate (2017 = 100)")
    if first_ntl > 0:
        ax.plot(merged["Year"], merged["NASA_Index_2017_100"], "-s", lw=2.4,
                label="NASA nighttime radiance (2017 = 100)")
    ax.set(title="EarthLens | Urban Estimate vs NASA Black Marble (2017–2025)",
           xlabel="Year", ylabel="Index (2017 = 100)")
    ax.set_xticks(range(2017, 2026))
    ax.grid(alpha=.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "EarthLens_NASA_Annual_Comparison.png", dpi=240)
    plt.close(fig)
    print("\nNASA annual summary:\n", nasa.to_string(index=False))
    print("\nComparison saved in:", OUT)
    print("IMPORTANT: Radiance is not built-up area; matching trends do not prove causation.")


def main():
    parser = argparse.ArgumentParser(description="EarthLens NASA Black Marble integration")
    parser.add_argument("--project", help="Your registered Google Earth Engine Cloud project ID")
    parser.add_argument("--roi", type=Path, help="OPTIONAL: original study-boundary GeoJSON (recommended)")
    parser.add_argument("--check", action="store_true", help="Offline input/area check only, no NASA download")
    args = parser.parse_args()
    complete_path = find_original_csv("Complete_Dataset")
    urban_path = find_original_csv("Urban_Growth_Analysis")
    complete = pd.read_csv(complete_path)
    urban = pd.read_csv(urban_path)
    if args.roi:
        geometry = get_geometry_object(args.roi)
        roi_label = "Original/custom polygon supplied by user"
    else:
        geometry = approx_polygon_from_samples(complete)
        roi_label = "APPROXIMATE sample convex hull; NOT original study-area boundary"
    (OUT / "EarthLens_Step8_Region_Used.geojson").write_text(
        json.dumps({"type": "Feature", "properties": {"source": roi_label},
                    "geometry": geometry}, indent=2), encoding="utf-8")
    print("Input Urban Growth:", urban_path.name)
    print("Input Complete Data:", complete_path.name)
    print("Study geometry:", roi_label)
    if args.check:
        print("Offline check PASSED: both datasets found, study polygon created.")
        print("No NASA data downloaded. Next run with --project YOUR_ID")
        return
    if not args.project:
        parser.error("NASA download requires --project YOUR_EARTH_ENGINE_PROJECT_ID (or run --check).")
    nasa = calculate_nasa_years(geometry, args.project)
    create_comparison(nasa, urban)


if __name__ == "__main__":
    main()