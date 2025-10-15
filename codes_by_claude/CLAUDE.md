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