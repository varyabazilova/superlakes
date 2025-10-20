import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

def plot_water_timeseries():
    """
    Create a time series plot with date on x-axis and two y-axes:
    - Left y-axis: Water area (km²)
    - Right y-axis: Feature count
    """
    
    # Read the data
    data_file = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/water_stats/water_summary_table_all_years.csv"
    
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    print(f"Loaded {len(df)} records from {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot water area on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Water Area (km²)', color=color1, fontsize=12)
    line1 = ax1.plot(df['date'], df['total_area_km2'], color=color1, marker='o', 
                     linewidth=2, markersize=6, label='Water Area')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for feature count
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Number of Water Features', color=color2, fontsize=12)
    line2 = ax2.plot(df['date'], df['feature_count'], color=color2, marker='s', 
                     linewidth=2, markersize=6, label='Feature Count')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add title and legend
    plt.title('Langtang Water Body Time Series (2021)', fontsize=14, fontweight='bold', pad=20)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Add text box with statistics
    stats_text = f"""Summary Statistics:
    Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
    Water Area: {df['total_area_km2'].min():.3f} - {df['total_area_km2'].max():.3f} km²
    Feature Count: {df['feature_count'].min()} - {df['feature_count'].max()}
    Average Area: {df['total_area_km2'].mean():.3f} km²
    Average Features: {df['feature_count'].mean():.1f}"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_file = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/codes_by_claude/water_timeseries_2021.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    # Print some insights
    print("\nKey Insights:")
    
    # Find peaks
    max_area_idx = df['total_area_km2'].idxmax()
    max_features_idx = df['feature_count'].idxmax()
    min_area_idx = df['total_area_km2'].idxmin()
    min_features_idx = df['feature_count'].idxmin()
    
    print(f"Peak water area: {df.loc[max_area_idx, 'total_area_km2']:.3f} km² on {df.loc[max_area_idx, 'date'].strftime('%Y-%m-%d')}")
    print(f"Peak feature count: {df.loc[max_features_idx, 'feature_count']} features on {df.loc[max_features_idx, 'date'].strftime('%Y-%m-%d')}")
    print(f"Minimum water area: {df.loc[min_area_idx, 'total_area_km2']:.3f} km² on {df.loc[min_area_idx, 'date'].strftime('%Y-%m-%d')}")
    print(f"Minimum feature count: {df.loc[min_features_idx, 'feature_count']} features on {df.loc[min_features_idx, 'date'].strftime('%Y-%m-%d')}")
    
    # Seasonal patterns
    df['month'] = df['date'].dt.month
    monthly_stats = df.groupby('month').agg({
        'total_area_km2': 'mean',
        'feature_count': 'mean'
    }).round(3)
    
    print(f"\nMonthly averages:")
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 9: 'Sep', 10: 'Oct'}
    for month in sorted(monthly_stats.index):
        month_name = month_names.get(month, f'Month {month}')
        print(f"{month_name}: {monthly_stats.loc[month, 'total_area_km2']:.3f} km², {monthly_stats.loc[month, 'feature_count']:.1f} features")

if __name__ == "__main__":
    plot_water_timeseries()