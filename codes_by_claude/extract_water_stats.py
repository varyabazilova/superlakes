import geopandas as gpd
import pandas as pd
import glob
import os
from datetime import datetime

def extract_stats_from_vector_files(vector_dir):
    """
    Extract total area, feature count, and date from vector polygon files
    
    Parameters:
    vector_dir (str): Directory containing vector polygon files
    
    Returns:
    pandas.DataFrame: Table with date, total_area_km2, feature_count
    """
    
    # Find all vector files
    vector_files = glob.glob(os.path.join(vector_dir, "*.gpkg"))
    vector_files.sort()
    
    if not vector_files:
        print(f"No vector files found in {vector_dir}")
        return None
    
    print(f"Found {len(vector_files)} vector files")
    
    results = []
    
    for vector_file in vector_files:
        filename = os.path.basename(vector_file)
        print(f"Processing {filename}...")
        
        try:
            # Read the vector file
            gdf = gpd.read_file(vector_file)
            
            # Extract date from filename (format: YYYY-MM-DD_water_polygons.gpkg)
            date_str = filename.split('_')[0]  # Gets "YYYY-MM-DD"
            date = pd.to_datetime(date_str)
            
            # Calculate statistics
            feature_count = len(gdf)
            total_area_km2 = gdf['area_km2'].sum() if 'area_km2' in gdf.columns else 0.0
            
            # If area_km2 column doesn't exist, calculate from geometry
            if 'area_km2' not in gdf.columns:
                if gdf.crs and not gdf.crs.is_geographic:
                    area_m2 = gdf.geometry.area.sum()
                else:
                    gdf_proj = gdf.to_crs('EPSG:32645')  # UTM Zone 45N
                    area_m2 = gdf_proj.geometry.area.sum()
                total_area_km2 = area_m2 / 1e6
            
            results.append({
                'date': date,
                'total_area_km2': total_area_km2,
                'feature_count': feature_count,
                'filename': filename
            })
            
            print(f"  Date: {date_str}, Features: {feature_count}, Area: {total_area_km2:.6f} km²")
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('date')
    
    return df

def create_water_summary_table(base_dir, years=['2020', '2021', '2022', '2023', '2024']):
    """
    Create summary table for multiple years
    
    Parameters:
    base_dir (str): Base directory containing water/langtangYYYY subdirectories
    years (list): List of years to process
    
    Returns:
    pandas.DataFrame: Combined summary table
    """
    
    all_data = []
    
    for year in years:
        vector_dir = os.path.join(base_dir, 'water', f'langtang{year}', 'vect')
        
        if os.path.exists(vector_dir):
            print(f"\nProcessing year {year}...")
            print("=" * 40)
            
            df_year = extract_stats_from_vector_files(vector_dir)
            
            if df_year is not None and not df_year.empty:
                df_year['year'] = year
                all_data.append(df_year)
        else:
            print(f"Vector directory not found: {vector_dir}")
    
    if all_data:
        # Combine all years
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values('date')
        
        # Format columns
        combined_df['date_str'] = combined_df['date'].dt.strftime('%Y-%m-%d')
        
        # Reorder columns
        output_df = combined_df[['date_str', 'total_area_km2', 'feature_count', 'year']].copy()
        output_df.columns = ['date', 'total_area_km2', 'feature_count', 'year']
        
        return output_df
    
    return None

def main():
    """
    Main function to create water statistics summary table
    """
    
    base_directory = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes"
    
    # Check if we have any vector directories
    available_years = []
    for year in ['2020', '2021', '2022', '2023', '2024']:
        vector_dir = os.path.join(base_directory, 'water', f'langtang{year}', 'vect')
        if os.path.exists(vector_dir):
            available_years.append(year)
    
    if not available_years:
        print("No vector directories found. You need to run vectorization first.")
        print("Currently only 2021 vectors exist in water/langtang2021/vect/")
        
        # Process only 2021 for now
        vector_dir_2021 = os.path.join(base_directory, 'water', 'langtang2021', 'vect')
        if os.path.exists(vector_dir_2021):
            print("\nProcessing 2021 data only...")
            df = extract_stats_from_vector_files(vector_dir_2021)
            
            if df is not None:
                # Add year column and format
                df['year'] = '2021'
                df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
                output_df = df[['date_str', 'total_area_km2', 'feature_count', 'year']].copy()
                output_df.columns = ['date', 'total_area_km2', 'feature_count', 'year']
                
                # Create output directory and save to CSV
                output_dir = os.path.join(base_directory, 'water_stats', 'langtang2021')
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, 'water_summary_table.csv')
                output_df.to_csv(output_file, index=False)
                
                print(f"\nSummary table saved to: {output_file}")
                print("\nPreview of the table:")
                print(output_df.to_string(index=False))
        return
    
    print(f"Found vector data for years: {available_years}")
    
    # Create summary table for all available years
    summary_df = create_water_summary_table(base_directory, available_years)
    
    if summary_df is not None:
        # Save to CSV - since only 2021 has data, save to langtang2021 folder
        if len(available_years) == 1 and '2021' in available_years:
            output_dir = os.path.join(base_directory, 'water_stats', 'langtang2021')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'water_summary_table.csv')
        else:
            output_dir = os.path.join(base_directory, 'water_stats')
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'water_summary_table_all_years.csv')
        summary_df.to_csv(output_file, index=False)
        
        print(f"\nSummary table saved to: {output_file}")
        print(f"Total records: {len(summary_df)}")
        
        # Show preview
        print("\nPreview of the table:")
        print(summary_df.head(10).to_string(index=False))
        
        print("\nSummary statistics:")
        print(f"Date range: {summary_df['date'].min()} to {summary_df['date'].max()}")
        print(f"Total area range: {summary_df['total_area_km2'].min():.4f} - {summary_df['total_area_km2'].max():.4f} km²")
        print(f"Feature count range: {summary_df['feature_count'].min()} - {summary_df['feature_count'].max()}")
        print(f"Average area: {summary_df['total_area_km2'].mean():.4f} km²")
        print(f"Average feature count: {summary_df['feature_count'].mean():.1f}")
        
        # Yearly summaries
        if len(available_years) > 1:
            print("\nYearly averages:")
            yearly_stats = summary_df.groupby('year').agg({
                'total_area_km2': ['mean', 'min', 'max', 'count'],
                'feature_count': ['mean', 'min', 'max']
            }).round(4)
            print(yearly_stats)
    
    else:
        print("No data found to process")

if __name__ == "__main__":
    main()