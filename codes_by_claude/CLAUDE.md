# Water Detection on Debris-Covered Ice - Unsupervised Methods

## Project Overview
Mapping water on debris-covered ice using Planet imagery (RGB + NIR bands) with unsupervised approaches.

## Suggested Algorithms

### 1. Threshold Selection (Unsupervised)
- **Otsu's method** - automatically finds optimal NDWI threshold
- **K-means clustering** on NDWI values (2-3 clusters: water, ice, debris)
- **Histogram analysis** - find natural breaks/valleys in NDWI distribution

### 2. Multi-band Unsupervised Clustering
- **K-means** on all 4 bands (RGB + NIR) simultaneously
- **Gaussian Mixture Models** for soft clustering
- **Spectral clustering** for non-linear separations

### 3. Hybrid Unsupervised Approaches
- Create NDWI mask → apply **watershed segmentation** → cluster segments
- **ISODATA clustering** (iterative self-organizing data analysis)
- **Mean-shift clustering** in spectral space

### 4. Smart Refinement
- Use **morphological operations** to clean up NDWI mask
- **Connected component analysis** - remove small isolated pixels
- **Texture-based filtering** - water typically has lower texture variance

### 5. Multi-criteria Unsupervised
```
Water_likelihood = f(NDWI, Blue_reflectance, Texture_homogeneity)
```

## Water Indices Reference
- **NDWI** (Normalized Difference Water Index): `(Green - NIR) / (Green + NIR)`
- **MNDWI** (Modified NDWI): `(Green - SWIR) / (Green + SWIR)` - requires SWIR band
- **AWEInsh** (Automated Water Extraction Index): combines multiple bands

## Implementation Notes
- Start with Otsu thresholding on NDWI for simplest approach
- Consider clustering in 4D spectral space for better debris-covered ice discrimination
- Texture analysis crucial for distinguishing water from surrounding materials

---

## Development Session Summary

### Initial Implementation
- Created `water_detection.py` for Planet imagery (RGB + NIR) analysis
- Implemented NDWI calculation: `(Green - NIR) / (Green + NIR)`
- Applied Otsu thresholding for unsupervised water detection

### Key Problem Identified
Initial run without glacier masking:
- **89.31% water detected** - suspiciously high
- **Otsu threshold: -0.166** - too permissive
- Issue: Including all terrain types (vegetation, rocks, shadows) skewed the threshold

### Solution: RGI Glacier Masking
Integrated RGI glacier shapefile to constrain analysis:
- **Input:** `/Users/varyabazilova/Desktop/glacial_lakes/RGI2000-v7-3/RGI2000-v7.0-C-15_south_asia_east.shp`
- **Glacier area:** 47.91% of image (11,554,409 pixels)
- **New results:** 51.76% water within glacier areas (much more realistic)
- **Improved threshold:** 0.011 (only considering glacier pixels)

### How Otsu Thresholding Works
1. **Histogram Analysis:** Creates histogram of NDWI values
2. **Class Separation:** Treats data as two classes (water vs non-water)
3. **Optimization:** Finds threshold that maximizes between-class variance
4. **Mathematical Process:**
   ```
   For each possible threshold t:
   - Calculate class weights w1, w2
   - Calculate class means μ1, μ2  
   - Between-class variance = w1 * w2 * (μ1 - μ2)²
   - Find t that maximizes this variance
   ```

### Enhanced Visualization
Added comprehensive analysis showing:
1. RGB composite and NDWI map
2. Glacier mask (RGI) overlay
3. NDWI histograms (all vs glacier pixels)
4. **Otsu optimization curve** - shows how algorithm finds optimal threshold
5. Water mask and RGB overlay
6. Class separation visualization
7. Detailed statistics

### Key Insight: Glacier-Constrained Analysis
**Critical difference between approaches:**
- **Without glacier mask:** Uses all 24.1M pixels → threshold = -0.166
- **With glacier mask:** Uses only glacier pixels → threshold = 0.011

The glacier-masked approach eliminates "noise" from non-glacial terrain and focuses on the actual problem: distinguishing water from ice/debris on glaciers.

### Output Files
- `water_mask_detailed.tif` - Binary water detection mask
- `water_mask_detailed_ndwi.tif` - NDWI values as TIFF
- `water_detection_results.png` - Comprehensive visualization with Otsu analysis

### Usage
```bash
python water_detection.py input_image.tif --glacier_shp glacier_polygons.shp --output water_mask.tif
```

---

## Extended Development Session - October 2025

### Problem with Otsu Thresholding
After initial implementation with Otsu, we discovered a critical issue:
- **Otsu struggles with ice/snow vs water distinction** on glaciers
- Ice, snow, and water can have very similar NDWI values
- Otsu assumes bimodal distribution but reality is more complex
- Results: 25-72% water detection (unrealistically high)

### Solution: Fixed Threshold Approach
Switched from automatic (Otsu) to **fixed threshold approach**:
- **Fixed threshold: NDWI > 0.2** for all dates
- Much more conservative and realistic results
- Consistent temporal comparison (same threshold across all dates)

### Temporal Data Processing
Created complete workflow for multi-temporal Planet imagery analysis:

#### 1. Mosaic Creation (`create_date_mosaics.py`)
- **Input:** Multiple Planet strips per date (15 strips → 8 dates)
- **Process:** Uses `rasterio.merge` to combine strips into single images per date
- **Output:** 8 temporal mosaics in `testimages25/` folder
- **Dates:** 2025-01-08, 2025-02-08, 2025-04-25, 2025-05-25, 2025-06-14, 2025-06-28, 2025-07-06, 2025-08-21

#### 2. NDWI Calculation (`calculate_ndwi.py`)
- **Input:** Planet 4-band imagery (RGB + NIR)
- **Formula:** NDWI = (Green - NIR) / (Green + NIR)
- **Output:** NDWI GeoTIFF files for each date
- **Location:** `/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/images/langtang2025/ndwi/`

#### 3. Water Detection (`map_water_from_ndwi.py`)
- **Input:** NDWI images
- **Process:** Fixed threshold (0.2) + glacier masking + optional morphological cleaning
- **Output:** Binary water masks + statistics CSV

### Morphological Cleaning Pipeline
Implemented 3-step cleaning process (optional):
1. **Opening** (disk radius=2): Remove scattered noise pixels
2. **Closing** (disk radius=3): Fill small gaps in water bodies
3. **Size filtering** (min 50 pixels): Remove tiny artifacts

**Cleaning efficiency:** 1-6% noise reduction, more effective in winter months with higher water detection.

### Results Comparison: Otsu vs Fixed Threshold

| Method | Threshold | Water % Range | Characteristics |
|--------|-----------|---------------|-----------------|
| **Otsu** | 0.011-0.076 (auto) | 25-72% | Too inclusive, picks up ice/snow |
| **Fixed** | 0.2 (constant) | 0.12-8.35% | Conservative, realistic water bodies |

### Fixed Threshold Results (Raw, No Cleaning)
**Temporal pattern (% of glacier area):**
- **Winter peak:** January (8.35%) - likely winter lakes/wet conditions
- **Spring transition:** February (3.69%)
- **Summer minimum:** April-August (0.12-0.48%) - consistent low water
- **Seasonal insight:** Opposite pattern from Otsu, more realistic

### RGB Temporal Composites
Developed visualization tools for temporal analysis:
- **maps.py:** Creates RGB composites from 3 different dates/bands
- **Color interpretation:** Each color channel represents water presence at different times
- **QGIS integration:** `save_rgb_composite()` creates GeoTIFF files for spatial analysis

**Color meaning example:**
- **White:** Water present in all 3 periods (persistent lakes)
- **Cyan:** Water in periods 2+3, frozen in period 1
- **Red:** Water only in early period (dried up later)

### Workflow Organization
Moved processing scripts to organized structure:

**📁 `image_processing/` folder:**
1. **`create_date_mosaics.py`** - Multi-strip → single mosaic per date
2. **`calculate_ndwi.py`** - Planet imagery → NDWI calculation
3. **`map_water_from_ndwi.py`** - NDWI → water masks (configurable threshold/cleaning)

### Key Files Created
- **8 temporal mosaics** (`testimages25/*.tif`)
- **16 NDWI files** (`outputs_fixed_raw/*_ndwi.tif`)
- **16 water masks** (`outputs_fixed_raw/*_water.tif`)
- **Temporal statistics** (`langtang_timeseries_fixed_raw.csv`)
- **RGB composites** (`analysis/testoutput/*.tif`) - ready for QGIS

### Technical Insights
1. **Fixed thresholds work better than Otsu** for glacier water detection
2. **Glacier masking is essential** - constrains analysis to relevant areas
3. **Morphological cleaning is optional** - provides 1-6% noise reduction
4. **Temporal composites reveal seasonal patterns** - winter peaks vs summer minimums
5. **3m Planet resolution** captures fine-scale supraglacial features

### Next Steps Available
- Batch process additional imagery when available
- Experiment with different threshold values (0.1, 0.15, 0.25, 0.3)
- Apply to other glacier regions
- Validate results against field observations
- Explore multi-temporal change detection analysis

### Command Quick Reference
```bash
# Step 1: Create mosaics
python image_processing/create_date_mosaics.py

# Step 2: Calculate NDWI
python image_processing/calculate_ndwi.py

# Step 3: Map water (raw threshold)
python image_processing/map_water_from_ndwi.py --threshold 0.2

# Step 3b: Map water with cleaning
python image_processing/map_water_from_ndwi.py --threshold 0.2 --cleaning
```

---

# Multi-Year Mosaic Creation - October 16, 2025

## Overview
Expanded analysis to create date-based mosaics from individual Planet scenes across multiple years (2020-2024). Main focus was processing non-composite datasets to generate one mosaic per acquisition date.

## Key Accomplishments

### 1. **Planet Download Issue Diagnosis**
- **Problem**: User ordered ~100 images but only got 2-3 files
- **Root Cause**: Planet's "composite by strip" merges multiple scenes from same date (not multiple dates)
- **Secondary Issue**: Large downloads (30GB) were failing/corrupting, resulting in `.download` files
- **Solution**: Found complete non-composite datasets for most years

### 2. **Mosaic Creation Pipeline**
Successfully created date-based mosaics from individual Planet scenes:

#### **2020**: ✅ **15 mosaics** from 91 individual scenes
- **Source**: `/Users/varyabazilova/Desktop/glacial_lakes/images/langtang2020_nocomp/`
- **Output**: 15 temporal mosaics (May-October 2020)
- **Coverage**: Excellent - dense fall coverage, good seasonal representation

#### **2021**: ✅ **11 mosaics** 
- **Source**: `/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/images_raw/langtang2021_nocomp/`
- **Output**: `/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/Images_mosaics/langtang2021mosaics/`
- **Coverage**: Winter through fall (Jan, Feb-Mar, Sep-Oct)

#### **2022**: ✅ **16 mosaics**
- **Source**: `/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/images_raw/langtang2022_nocomp/`
- **Output**: `/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/Images_mosaics/langtang2022mosaics/`
- **Coverage**: Excellent spring/summer coverage (April-July)

#### **2023**: ⚠️ **12 mosaics** (incomplete)
- **Source**: `/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/images_raw/langtang2023_nocomp/`
- **Output**: `/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/Images_mosaics/langtang2023mosaics/`
- **Issue**: Only 19 images total → **incomplete download**

#### **2024**: ⚠️ **9 mosaics** (incomplete)
- **Source**: `/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/images_raw/langtang2024_nocomp_strange/`
- **Output**: `/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/Images_mosaics/langtang2024mosaics/`
- **Issue**: Only 15 images total (expected ~87) → **severely incomplete download**

### 3. **Code Improvements**
- **Enhanced date extraction**: Updated `create_date_mosaics.py` to handle both YYYY-MM-DD and YYYYMMDD formats
- **Added .gitignore rule**: `images*/` to ignore all folders starting with "images"
- **Corruption handling**: Script now handles corrupted TIFF files gracefully

## Data Quality Assessment

| Year | Images Found | Dates Created | Status | Coverage Quality |
|------|-------------|---------------|---------|------------------|
| 2020 | 91 | 15 | ✅ Complete | Excellent |
| 2021 | ~50+ | 11 | ✅ Complete | Good |
| 2022 | ~60+ | 16 | ✅ Complete | Excellent |
| 2023 | 19 | 12 | ❌ Incomplete | Poor |
| 2024 | 15 | 9 | ❌ Incomplete | Poor |

## Current Issues & Next Steps

### **Immediate Actions Needed**
1. **Re-download 2023 & 2024 datasets** - current downloads are severely incomplete
2. **Use Planet CLI for reliable downloads** - provided example with Planet Scene IDs:
   ```bash
   planet data download --item-type PSScene --asset-type analytic_sr_udm2 --dest ./downloads --id-list ids.txt
   ```

### **Ready for Analysis**
- **2020-2022**: 42 date-based mosaics ready for water detection analysis
- **Covers**: 3 years of temporal data with good seasonal representation
- **Next step**: Run water detection pipeline on these complete datasets

### **File Organization**
- **Raw images**: `images_raw/langtangYYYY_nocomp/`
- **Mosaics**: `Images_mosaics/langtangYYYYmosaics/`
- **Previous work**: CLAUDE.md documents complete 2025 analysis with fixed threshold approach

## Technical Notes
- **Mosaic script**: `image_processing/create_date_mosaics.py` (handles both individual scenes and strips)
- **File pattern**: `*_3B_AnalyticMS_SR_harmonized_clip.tif` for individual scenes
- **Corruption issues**: Some Planet files have TIFF read errors, script skips these automatically

---
**Status**: 2020-2022 datasets complete and ready for water detection analysis. 2023-2024 require re-download to proceed.

---

# Water Detection with Slope Masking - October 20, 2025

## Current Session Summary

### Objective
Run water detection analysis on all 83 NDWI files (2020-2024) using slope-based masking instead of glacier masking for more targeted analysis of potential supraglacial lake locations.

### Key Modifications Made

#### 1. **Image Availability Visualization**
Created `plot_image_availability.py` with enhanced timeline visualization:
- **X-axis**: Months (Jan-Dec) - all dates aligned regardless of year
- **Y-axis**: Years (2020-2024) - each year gets separate horizontal row
- **Features**: Monthly and half-monthly grid lines, hollow circle markers, statistics box
- **Output**: Visual timeline showing seasonal patterns across all years

#### 2. **Water Detection Script Updates**
Modified `map_water_from_ndwi.py` to use slope-based analysis mask:
- **Original**: RGI glacier mask (`/Users/varyabazilova/Desktop/glacial_lakes/RGI2000-v7-3/RGI2000-v7.0-C-15_south_asia_east.shp`)
- **Updated**: Slope mask (`/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/other_data/vect/langtang_slope_lte10_mask_utm_vect_fix.shp`)

**Function changes:**
- `create_glacier_mask()` → `create_analysis_mask()` - generic shapefile masking
- Updated all variable names: `glacier_mask` → `analysis_mask`
- Statistics: `water_pct_of_glacier` → `water_pct_of_analysis`
- Default parameters now point to slope shapefile

#### 3. **Analysis Setup**
**Input data structure:**
```
ndwi/
├── langtang2020/ (15 NDWI files)
├── langtang2021/ (11 NDWI files)  
├── langtang2022/ (16 NDWI files)
├── langtang2023/ (20 NDWI files)
└── langtang2024/ (21 NDWI files)
Total: 83 NDWI files
```

**Output structure:**
```
water/
├── langtang2020/ (water masks + statistics)
├── langtang2021/ (water masks + statistics)
├── langtang2022/ (water masks + statistics)
├── langtang2023/ (water masks + statistics)
└── langtang2024/ (water masks + statistics)
```

### Processing Pipeline
1. **NDWI Threshold**: Fixed threshold of 0.2 (conservative, based on previous analysis)
2. **Slope Masking**: Restrict analysis to areas with slope ≤10° (potential lake locations)
3. **Water Detection**: Apply threshold within slope-masked areas only
4. **Output**: Binary water masks (.tif) + temporal statistics (.csv) for each year

### Rationale for Slope Masking
- **Supraglacial lakes** typically form in **low-slope areas** (≤10°)
- **Steeper terrain** less likely to retain water bodies
- **More targeted analysis** compared to broad glacier masking
- **Reduces false positives** from ice/snow confusion in steep areas

### Commands Ready to Execute
```bash
# 2020
python image_processing/map_water_from_ndwi.py --ndwi_dir ndwi/langtang2020 --output_dir water/langtang2020 --threshold 0.2

# 2021
python image_processing/map_water_from_ndwi.py --ndwi_dir ndwi/langtang2021 --output_dir water/langtang2021 --threshold 0.2

# 2022
python image_processing/map_water_from_ndwi.py --ndwi_dir ndwi/langtang2022 --output_dir water/langtang2022 --threshold 0.2

# 2023
python image_processing/map_water_from_ndwi.py --ndwi_dir ndwi/langtang2023 --output_dir water/langtang2023 --threshold 0.2

# 2024
python image_processing/map_water_from_ndwi.py --ndwi_dir ndwi/langtang2024 --output_dir water/langtang2024 --threshold 0.2
```

### Expected Outputs
For each year:
- **Water masks**: Binary GeoTIFF files showing detected water pixels
- **Statistics CSV**: Temporal analysis with water percentages, areas, NDWI statistics
- **Analysis metrics**: Water detection within slope-constrained areas only

### Next Steps After Processing
1. **Temporal analysis**: Compare water detection across years and seasons
2. **Validation**: Visual inspection of water masks against RGB imagery
3. **Threshold sensitivity**: Test different NDWI thresholds (0.1, 0.15, 0.25, 0.3)
4. **Change detection**: Identify persistent vs. ephemeral water bodies
5. **Export for GIS**: Load results into QGIS for spatial analysis

---
**Status**: Ready to execute water detection on 83 NDWI files with slope-based masking. Bash tool connectivity issues prevented execution - manual terminal run required.

---

# Water Detection Automation & Vectorization - October 20, 2025

## Session Accomplishments

### 1. **Corrected Water Detection Threshold Issue**
**Problem Identified**: Previous Otsu-based water detection was massively over-classifying water:
- **Otsu thresholds**: 0.075, 0.022, even negative values (-0.005)
- **Result**: 28-76% of analysis area classified as water (unrealistic)
- **Cause**: Otsu algorithm finding very low thresholds due to ice/snow/water similarity

**Solution**: Switched to **fixed NDWI threshold approach**:
- **New threshold**: NDWI > 0.0 (simple but effective)
- **Rationale**: NDWI > 0 = water, NDWI < 0 = not water
- **Results**: Much more realistic 3-24% water coverage

### 2. **Automated Water Detection for All Years (2020-2024)**
Successfully processed **83 NDWI files** across 5 years with corrected threshold:

#### **Processing Results:**
- **2020**: 15 images → 0.48% - 23.97% water coverage (avg: 10.18%)
- **2021**: 11 images → 3.40% - 11.83% water coverage (avg: 7.31%)
- **2022**: 16 images → 2.80% - 11.21% water coverage (avg: 5.09%)
- **2023**: 20 images → 3.14% - 20.52% water coverage (avg: 6.82%)
- **2024**: 21 images → 2.12% - 7.54% water coverage (avg: 4.52%)

**Key Improvements:**
- **Fixed nodata handling**: Updated script to prevent NaN errors in output TIFFs
- **Consistent processing**: Same threshold (0.0) applied across all years
- **Proper file organization**: All outputs saved to `water/langtangYYYY/` directories

### 3. **Vector Polygon Creation & Area Analysis**
Developed GIS-style vector analysis workflow:

#### **Raster-to-Vector Conversion**
Created `batch_vectorize_2021.py` to convert binary water masks to vector polygons:
- **Process**: Convert raster pixels → vector polygons (like GIS "raster to vector")
- **Output**: Individual `.gpkg` files for each date with water body polygons
- **Location**: `water/langtang2021/vect/`

#### **2021 Vectorization Results:**
- **Total water features**: 1,605 individual water bodies across all dates
- **Total water area**: 11.33 km² cumulative across all dates
- **Feature count range**: 78 - 251 water bodies per date
- **Largest single feature**: Up to 1.67 km² on peak dates

#### **Individual Date Breakdown:**
```
2021-01-28: 251 features, 1.665 km²  (winter peak)
2021-02-24: 188 features, 1.432 km²
2021-03-03: 231 features, 1.486 km²
2021-03-17: 119 features, 0.912 km²
2021-09-04: 147 features, 1.216 km²
2021-09-18: 109 features, 0.672 km²
2021-09-20: 138 features, 0.899 km²
2021-10-05: 78 features,  0.479 km²  (minimum)
2021-10-09: 125 features, 0.930 km²
2021-10-13: 97 features,  0.707 km²
2021-10-15: 122 features, 0.930 km²
```

### 4. **Statistical Summary Table Creation**
Developed `extract_water_stats.py` for tabular analysis:
- **Reads vector polygons** → extracts total area, feature count, date
- **Output format**: CSV with columns: `date`, `total_area_km2`, `feature_count`, `year`
- **Purpose**: Ready for plotting, statistical analysis, Excel import

#### **Output Table Structure:**
```csv
date,total_area_km2,feature_count,year
2021-01-28,1.665333,251,2021
2021-02-24,1.432125,188,2021
2021-03-03,1.485900,231,2021
...
```

### 5. **Enhanced Analysis Capabilities**
Created comprehensive toolkit for water body analysis:

#### **Scripts Created:**
1. **`test_water_area.py`**: Single-file vector analysis and area calculation
2. **`batch_vectorize_2021.py`**: Batch raster-to-vector conversion
3. **`extract_water_stats.py`**: Summary statistics extraction
4. **`plot_water_timeseries.py`**: Time series visualization (dual y-axis plot)

#### **GIS-Style Workflow:**
1. **Raster → Vector**: Convert binary masks to polygons
2. **Zonal Statistics**: Calculate area and count for each polygon
3. **Temporal Analysis**: Track changes over time
4. **Visualization**: Plot trends and patterns

### 6. **Technical Achievements**

#### **Area Calculation Methods:**
- **Projected coordinates**: Direct geometry.area calculation
- **Geographic coordinates**: Automatic reprojection to UTM Zone 45N
- **Validation**: Areas match between pixel counting and vector methods

#### **Data Quality Improvements:**
- **Morphological cleaning**: 3-step noise reduction (opening, closing, size filtering)
- **Minimum feature size**: 50 pixels (removes tiny artifacts)
- **Consistent projections**: All outputs maintain proper coordinate systems

#### **File Formats:**
- **Water masks**: GeoTIFF (.tif) with projection data
- **Vector polygons**: GeoPackage (.gpkg) - modern, efficient format
- **Statistics**: CSV for easy import into analysis software

### 7. **Key Insights from Analysis**

#### **Temporal Patterns (2021):**
- **Winter peak**: January shows highest water coverage (1.67 km²)
- **Seasonal variation**: 3.5x difference between min/max water coverage
- **Feature dynamics**: More features in winter (251) vs fall (78)

#### **Methodological Success:**
- **Threshold correction**: NDWI > 0.0 produces realistic results
- **Vector validation**: Polygon areas confirm raster pixel counting
- **Automation**: Pipeline processes 83 files efficiently

### Current Status & Next Steps

#### **Ready for Analysis:**
- **2021 fully processed**: Rasters + vectors + statistics table complete
- **2020, 2022-2024**: Water masks generated, ready for vectorization
- **Complete temporal dataset**: 83 observations across 5 years

#### **Available for Immediate Use:**
1. **Water masks**: Binary detection results for all years
2. **Vector polygons**: Individual water body analysis (2021)
3. **Statistics table**: Temporal trends and quantitative analysis
4. **Visualization tools**: Time series plotting capabilities

#### **Expansion Possibilities:**
- Vectorize remaining years (2020, 2022-2024)
- Multi-year comparative analysis
- Threshold sensitivity testing
- Validation against field observations
- Integration with climate/meteorological data

---
**Current Status**: Water detection automated and validated. Vector analysis pipeline established. 2021 fully processed with quantitative results ready for scientific analysis.

---

# 2025 Planet Imagery Processing - October 23, 2025

## Session Accomplishments

### 1. **Complete 2025 Dataset Processing**
Successfully processed all 2025 Planet imagery through the full pipeline:

#### **Mosaic Creation Results:**
- **Input**: 36 Planet scene IDs from 3 separate orders (Jan-Feb, May-Sept, and harmonized datasets)
- **Output**: 21 temporal mosaics covering January through September 2025
- **Location**: `/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/Images_mosaics/langtang2025_harmonized_mosaics/`
- **Date Coverage**: 2025-01-20 through 2025-09-05

#### **NDWI Calculation:**
- **Processed**: All 21 mosaics → NDWI calculation
- **Formula**: (Green - NIR) / (Green + NIR)
- **Output**: 21 NDWI files saved to `/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/ndwi/langtang2025/`
- **NDWI Range**: -0.863 to +0.999 across all dates

#### **Water Detection Results:**
Applied fixed threshold approach (NDWI > 0.0) with slope-based masking:

**Temporal Water Coverage Pattern (% of analysis area):**
- **Winter peak (Jan-Feb)**: 16.29% - 17.86% (highest water detection)
- **Spring transition (Apr)**: 11.10% - 12.79% 
- **Late spring (May)**: 5.89% - 9.29%
- **Summer minimum (Jun-Jul)**: 4.53% - 5.37% (lowest water detection)
- **Early fall (Sep)**: 4.90%

**Key Statistics:**
- **Maximum water**: 17.86% on 2025-01-30 (279,396 pixels)
- **Minimum water**: 4.53% on 2025-06-28 (70,854 pixels)
- **Average coverage**: 10.20% across all dates
- **Seasonal variation**: ~4x difference between winter peak and summer minimum

### 2. **Multi-Year Dataset Completion**
Now have complete processing pipeline applied to 6 years of Planet imagery:

| Year | Dates Processed | Water Masks | Status |
|------|----------------|-------------|---------|
| 2020 | 15 | ✅ | Complete |
| 2021 | 11 | ✅ | Complete + Vectorized |
| 2022 | 16 | ✅ | Complete |
| 2023 | 20 | ✅ | Complete |
| 2024 | 21 | ✅ | Complete |
| **2025** | **21** | **✅** | **Complete** |
| **Total** | **104** | **✅** | **All years processed** |

### 3. **Technical Pipeline Validation**
The 2025 processing confirms the established methodology:

#### **Threshold Effectiveness:**
- **NDWI > 0.0**: Produces realistic water coverage (4.5% - 18%)
- **Seasonal pattern**: Winter peaks, summer minimums (consistent with glacial lake dynamics)
- **No over-classification**: Unlike previous Otsu approach that yielded 25-76% coverage

#### **Processing Efficiency:**
- **Mosaic creation**: Handles multiple Planet strips per date automatically
- **NDWI calculation**: Batch processes all temporal mosaics
- **Water detection**: Applies consistent threshold with slope-based masking
- **Output quality**: All files maintain proper projection and nodata handling

### 4. **Dataset Characteristics - 2025**
**Temporal Distribution:**
- **Winter concentration**: 6 dates (Jan-Feb) with highest water detection
- **Spring coverage**: 4 dates (Apr-May) with moderate detection  
- **Summer coverage**: 10 dates (May-Jul) with variable detection
- **Fall coverage**: 1 date (Sep) with low detection

**Data Quality:**
- **Consistent resolution**: All mosaics at 3m Planet resolution
- **Proper georeferencing**: UTM Zone 45N projection maintained
- **Complete coverage**: No missing dates due to processing failures
- **Range validation**: NDWI values within expected -1 to +1 range

### 5. **Analysis-Ready Outputs**
**For 2025 data:**
- **Water masks**: 21 binary GeoTIFF files ready for GIS analysis
- **Statistics**: Temporal trends saved to CSV format
- **NDWI rasters**: Available for threshold sensitivity testing
- **Mosaics**: Original imagery available for visual validation

### Next Steps Available
1. **Vector polygon creation** for 2025 (following 2021 methodology)
2. **Multi-year comparative analysis** across all 6 years (2020-2025)
3. **Threshold sensitivity analysis** for 2025 data
4. **Seasonal pattern analysis** using complete temporal dataset
5. **Climate correlation** with complete 6-year record

### File Organization Summary
```
2025 Processing Pipeline:
├── Raw Planet scenes (36 IDs across 3 orders)
├── Images_mosaics/langtang2025_harmonized_mosaics/ (21 mosaics)
├── ndwi/langtang2025/ (21 NDWI files)
└── water/langtang2025/ (21 water masks + statistics)
```

---
**Current Status**: Complete 6-year Planet imagery dataset (2020-2025) processed with water detection pipeline. 104 total temporal observations ready for comprehensive glacial lake analysis.

---

# DINOv3 Foundational Model Exploration - October 28, 2025

## Problem with Current Approach
Fixed NDWI > 0.0 threshold works well on some images but fails on others, creating inconsistent water detection across the temporal dataset. Automation is needed for processing large numbers of images without manual threshold adjustment per image.

## DINOv3 Potential Solution

### What is DINOv3?
- **Foundational vision model** from Meta with 7B parameters
- **Self-supervised learning** on 1.7B images 
- **Dense feature extraction** - produces high-resolution features for every pixel
- **Satellite/aerial imagery strength** - specifically mentioned as application area
- **Off-the-shelf usage** - no training required for feature extraction

### Proposed Workflow: Manual Labeling + DINOv3

#### **Step 1: One-Time Manual Labeling**
- Select one representative image from 2025 dataset
- Manually draw water body polygons in QGIS
- Create binary mask: 1 = water, 0 = not-water

#### **Step 2: Feature Extraction**
- Run pre-trained DINOv3 on labeled image
- Extract dense features for every pixel
- Combine features with manual labels

#### **Step 3: Train Simple Classifier**
- Use basic classifier (Random Forest/small neural network)
- **Input**: DINOv3 features (+ optionally NDWI values)
- **Output**: water (1) or not-water (0) prediction
- Train on single labeled image

#### **Step 4: Automated Application**
- Run DINOv3 feature extraction on all 104 images (2020-2025)
- Apply trained classifier for automated water detection
- Consistent results across all temporal observations

### Advantages Over Fixed Thresholds
- **Visual pattern recognition**: Learns texture, context, spatial relationships
- **Handles variations**: Seasonal lighting, atmospheric conditions, ice/snow confusion
- **Automation**: One manual labeling session → automated processing of entire dataset
- **Combines spectral + visual**: Can use both DINOv3 features AND NDWI values
- **Robust**: Foundation model trained on massive diverse datasets

### Technical Considerations
- **Input format**: Use false color composites (NIR-Red-Green) as RGB input to DINOv3
- **Feature combination**: Hybrid approach using DINOv3 features + NDWI spectral values
- **Computational**: More intensive than simple thresholding but manageable for automation

### Next Steps (When Ready)
1. Select best representative image from 2025 dataset for manual labeling
2. Set up DINOv3 environment and model loading
3. Create manual water body labels
4. Implement feature extraction + classifier training pipeline
5. Test on subset of images before full dataset application

---
**Status**: DINOv3 approach identified as potential solution for automated, consistent water detection across large temporal datasets. Ready for implementation when needed.

---

# DINOv3 Implementation Setup - October 29, 2025

## Environment Setup Complete ✅

### Initial Problems Encountered
- **Fresh superlakes environment**: No packages installed initially
- **PyTorch installation issues**: M1 Mac compatibility problems
- **Corrupted environment**: pip thought it was in python3.1 directory while running python3.13
- **Solution**: Recreated clean environment with proper Python 3.11

### Successful Installation Steps
```bash
# Create fresh environment
conda deactivate
conda remove --name superlakes --all
conda create --name superlakes python=3.11
conda activate superlakes

# Install packages
conda install pytorch torchvision -c conda-forge
pip install jupyterlab transformers scikit-learn
conda install nb_conda_kernels -c conda-forge
```

### Environment Verification
- **Python**: 3.11.13 ✅
- **PyTorch**: Installed via conda-forge ✅
- **JupyterLab**: Installed ✅
- **Transformers**: For DINOv2 model loading ✅
- **nb_conda_kernels**: For Jupyter environment management ✅

## Next Steps Ready
1. **Start JupyterLab**: `jupyter lab`
2. **Test DINOv2 loading** in notebook
3. **Select representative 2025 image** for manual labeling
4. **Begin feature extraction pipeline**

### Key Learnings
- **M1 Mac**: Use conda-forge channel for better ARM64 support
- **Environment corruption**: Fresh recreation faster than debugging
- **Package management**: Mix conda (for core packages) + pip (for Python-specific packages)

---
**Status**: Development environment ready for DINOv3 experimentation. All dependencies installed and verified.