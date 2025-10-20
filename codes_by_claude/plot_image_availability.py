#!/usr/bin/env python3
"""
Visualize image availability across time (2020-2024)
Shows timeline with monthly divisions and dots for available image dates
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import glob
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

def extract_dates_from_mosaics(base_dir):
    """
    Extract dates from all mosaic files across all years
    
    Returns:
    list: List of datetime objects for all available images
    """
    all_dates = []
    
    # Define year directories
    year_dirs = {
        2020: "langtang2020_harmonized_mosaics",
        2021: "langtang2021_harmonized_mosaics", 
        2022: "langtang2022_harmonized_mosaics",
        2023: "langtang2023_harmonized_mosaics",
        2024: "langtang2024_harmonized_mosaics"
    }
    
    for year, dir_name in year_dirs.items():
        year_path = os.path.join(base_dir, dir_name)
        
        if os.path.exists(year_path):
            # Find all mosaic files
            mosaic_files = glob.glob(os.path.join(year_path, "*_composite_mosaic.tif"))
            
            for file_path in mosaic_files:
                basename = os.path.basename(file_path)
                # Extract date pattern YYYY-MM-DD
                match = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
                if match:
                    date_str = match.group(1)
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        all_dates.append(date_obj)
                        print(f"Found image: {date_str}")
                    except ValueError:
                        print(f"Could not parse date: {date_str}")
        else:
            print(f"Directory not found: {year_path}")
    
    return sorted(all_dates)

def create_availability_plot(dates, output_path=None):
    """
    Create timeline plot showing image availability with separate lines for each year
    
    Parameters:
    dates (list): List of datetime objects
    output_path (str): Optional path to save the plot
    """
    if not dates:
        print("No dates found to plot")
        return
    
    # Organize dates by year
    dates_by_year = {}
    for date in dates:
        year = date.year
        if year not in dates_by_year:
            dates_by_year[year] = []
        dates_by_year[year].append(date)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Define years and colors
    years = sorted(dates_by_year.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for each year
    
    # Create monthly and half-monthly grid lines (vertical)
    for month in range(1, 13):
        # First day of month (major grid line)
        month_date = datetime(2020, month, 1)
        ax.axvline(x=month_date, color='lightgray', linestyle='-', alpha=0.5, linewidth=1)
        
        # 15th of month (half-month grid line)
        half_month_date = datetime(2020, month, 15)
        ax.axvline(x=half_month_date, color='lightgray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Plot dots for each year
    for i, year in enumerate(years):
        year_dates = dates_by_year[year]
        
        # Convert dates to month-day format (remove year component for x-axis)
        month_day_dates = []
        for date in year_dates:
            # Convert to same reference year (2020) to align on x-axis
            aligned_date = datetime(2020, date.month, date.day)
            month_day_dates.append(aligned_date)
        
        # Plot dots at year position on y-axis
        ax.scatter(month_day_dates, [year] * len(month_day_dates), 
                  facecolors='none', edgecolors=colors[i % len(colors)], 
                  s=120, alpha=0.8, linewidths=2, zorder=3, 
                  label=f'{year} ({len(year_dates)} images)')
    
    # Set axis limits
    ax.set_xlim(datetime(2020, 1, 1), datetime(2020, 12, 31))
    ax.set_ylim(min(years) - 0.5, max(years) + 0.5)
    
    # Format x-axis (months)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Set y-axis (years)
    ax.set_yticks(years)
    ax.set_yticklabels([str(year) for year in years])
    ax.set_ylabel('Year', fontsize=12, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_title('Planet Imagery Availability Timeline - Langtang Glacial Lakes\n'
                'Each row shows one year, dots show available images by month', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3)
    ax.grid(True, axis='y', alpha=0.2)
    
    # Add statistics text
    stats_text = f"Total images: {len(dates)}\n"
    stats_text += f"Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}\n\n"
    
    stats_text += "Images per year:\n"
    for year in years:
        count = len(dates_by_year[year])
        stats_text += f"  {year}: {count} images\n"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top', horizontalalignment='right', fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax

def main():
    """Main function"""
    print("Creating Planet imagery availability timeline...")
    print("=" * 60)
    
    # Configuration
    base_mosaic_dir = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/Images_mosaics"
    output_path = "/Users/varyabazilova/Desktop/glacial_lakes/super_lakes/codes_by_claude/imagery_availability_timeline.png"
    
    # Extract dates
    print("Extracting dates from mosaic files...")
    dates = extract_dates_from_mosaics(base_mosaic_dir)
    
    print(f"\nFound {len(dates)} images total")
    print("Date range:", min(dates).strftime('%Y-%m-%d'), "to", max(dates).strftime('%Y-%m-%d'))
    
    # Create plot
    print("\nCreating timeline plot...")
    fig, ax = create_availability_plot(dates, output_path)
    
    print("\nTimeline visualization complete!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()