import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import isodate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COUNTRIES = {
    "FR": "France",
    "US": "United States", 
    "IN": "India"
}

DATA_FILES = {
    "FR": ("youtube_mostpopular_fr.json", "ytb_categories_fr.json"),
    "US": ("youtube_mostpopular_us.json", "ytb_categories_us.json"),
    "IN": ("youtube_mostpopular_in.json", "ytb_categories_in.json")
}

class YouTubeAnalyzer:
    """A class to analyze YouTube most popular videos data across countries."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.category_maps: Dict[str, Dict[str, str]] = {}
    
    def load_json(self, path: str) -> Dict[str, Any]:
        """Load JSON data from file with error handling."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return {}
    
    def safe_int_conversion(self, value: Any) -> int:
        """Safely convert value to integer, return NaN if conversion fails."""
        try:
            return int(value) if value is not None else np.nan
        except (ValueError, TypeError):
            return np.nan
    
    def parse_duration(self, duration: str) -> float:
        """Parse ISO 8601 duration to seconds."""
        try:
            if not duration:
                return np.nan
            return isodate.parse_duration(duration).total_seconds()
        except Exception:
            return np.nan
    
    def parse_datetime(self, datetime_str: str) -> pd.Timestamp:
        """Parse datetime string to pandas Timestamp."""
        try:
            return pd.to_datetime(datetime_str) if datetime_str else pd.NaT
        except Exception:
            return pd.NaT
    
    def extract_video_data(self, data: Dict[str, Any], country: str) -> List[Dict[str, Any]]:
        """Extract video data from JSON and convert to list of dictionaries."""
        rows = []
        items = data.get("items", [])
        
        if not items:
            logger.warning(f"No items found for country: {country}")
        
        for item in items:
            try:
                # Extract nested data
                snippet = item.get("snippet", {})
                statistics = item.get("statistics", {})
                content_details = item.get("contentDetails", {})
                
                # Parse and clean data
                published_at = self.parse_datetime(snippet.get("publishedAt"))
                duration_s = self.parse_duration(content_details.get("duration"))
                tags = snippet.get("tags") or []
                
                row = {
                    "country": country,
                    "videoId": item.get("id"),
                    "title": snippet.get("title"),
                    "channelTitle": snippet.get("channelTitle"),
                    "categoryId": str(snippet.get("categoryId", "")),
                    "publishedAt": published_at,
                    "duration_s": duration_s,
                    "viewCount": self.safe_int_conversion(statistics.get("viewCount")),
                    "likeCount": self.safe_int_conversion(statistics.get("likeCount")),
                    "commentCount": self.safe_int_conversion(statistics.get("commentCount")),
                    "tags_count": len(tags)
                }
                rows.append(row)
                
            except Exception as e:
                logger.error(f"Error processing item {item.get('id', 'unknown')}: {e}")
                continue
        
        return rows
    
    def load_category_mapping(self, categories_data: Dict[str, Any]) -> Dict[str, str]:
        """Create mapping from category ID to category name."""
        category_map = {}
        for item in categories_data.get("items", []):
            try:
                category_id = str(item["id"])
                category_name = item["snippet"]["title"]
                category_map[category_id] = category_name
            except KeyError as e:
                logger.error(f"Missing key in category data: {e}")
                continue
        
        return category_map
    
    def load_all_data(self) -> pd.DataFrame:
        """Load and process all YouTube data files."""
        all_rows = []
        
        for country_code, (data_file, categories_file) in DATA_FILES.items():
            logger.info(f"Processing data for {COUNTRIES[country_code]}")
            
            # Load video data
            video_data = self.load_json(data_file)
            if not video_data:
                logger.warning(f"No data loaded for {country_code}")
                continue
            
            # Load category data
            categories_data = self.load_json(categories_file)
            self.category_maps[country_code] = self.load_category_mapping(categories_data)
            
            # Extract video rows
            country_rows = self.extract_video_data(video_data, country_code)
            all_rows.extend(country_rows)
        
        # Create DataFrame
        df = pd.DataFrame(all_rows)
        
        if df.empty:
            logger.error("No data was loaded successfully")
            return df
        
        # Data cleaning and preprocessing
        df = self.clean_data(df)
        
        # Add category names
        df["categoryName"] = df.apply(self.map_category_name, axis=1)
        
        self.df = df
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the DataFrame."""
        # Fill missing view counts with 0
        df["viewCount"] = df["viewCount"].fillna(0)
        
        # Convert to appropriate data types
        numeric_columns = ["viewCount", "likeCount", "commentCount", "duration_s", "tags_count"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing essential data
        df = df.dropna(subset=["videoId", "country"])
        
        logger.info(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def map_category_name(self, row: pd.Series) -> str:
        """Map category ID to category name for a given row."""
        country = row["country"]
        category_id = row["categoryId"]
        
        category_map = self.category_maps.get(country, {})
        return category_map.get(category_id, "Unknown")
    
    def get_top_categories_by_country(self, top_n: int = 10) -> Dict[str, pd.Series]:
        """Get top N categories by total view count for each country."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        top_categories = {}
        for country_code in COUNTRIES.keys():
            country_data = self.df[self.df["country"] == country_code]
            if not country_data.empty:
                categories = (country_data.groupby("categoryName")["viewCount"]
                            .sum()
                            .sort_values(ascending=False)
                            .head(top_n))
                top_categories[country_code] = categories
        
        return top_categories
    
    def plot_category_analysis(self, top_n: int = 10, figsize: Tuple[int, int] = (18, 6)) -> None:
        """Create horizontal bar plots showing top categories by view count for each country."""
        top_categories = self.get_top_categories_by_country(top_n)
        
        if not top_categories:
            logger.warning("No category data available for plotting")
            return
        
        # Set up the plot style
        plt.style.use('default')
        fig, axes = plt.subplots(1, len(top_categories), figsize=figsize)
        
        # Handle single subplot case
        if len(top_categories) == 1:
            axes = [axes]
        
        colors = sns.color_palette("husl", len(top_categories))
        
        for idx, (country_code, categories) in enumerate(top_categories.items()):
            ax = axes[idx]
            
            # Create horizontal bar plot (reversed to show highest at top)
            y_pos = range(len(categories))
            bars = ax.barh(y_pos, categories.values, color=colors[idx], alpha=0.8)
            
            # Customize the plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories.index, fontsize=10)
            ax.set_xlabel("Total View Count", fontsize=12)
            ax.set_title(f"{COUNTRIES[country_code]}", fontsize=14, fontweight='bold')
            ax.invert_yaxis()  # Highest values at top
            
            # Format x-axis labels for better readability
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            
            # Add value labels on bars
            for bar, value in zip(bars, categories.values):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{value:,.0f}', ha='left', va='center', 
                       fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # lower the axes area so title sits inside visible region
        plt.suptitle("Top Video Categories by Total View Count", fontsize=16, fontweight='bold', y=0.99)
        plt.show()
    
    def plot_country_statistics(self, figsize: Tuple[int, int] = (12, 6)) -> None:
        """Create bar plots showing average statistics by country."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        # Calculate statistics by country
        country_stats = self.df.groupby("country").agg({
            "viewCount": "mean",
            "likeCount": "mean", 
            "commentCount": "mean",
            "duration_s": "mean"
        }).round(0)
        
        # Map country codes to full names
        country_stats.index = [COUNTRIES.get(code, code) for code in country_stats.index]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Average Video Statistics by Country", fontsize=16, fontweight='bold')
        
        stats_config = [
            ("viewCount", "Average View Count", axes[0,0]),
            ("likeCount", "Average Like Count", axes[0,1]),
            ("commentCount", "Average Comment Count", axes[1,0]),
            ("duration_s", "Average Duration (seconds)", axes[1,1])
        ]
        
        colors = sns.color_palette("Set2", len(country_stats))
        
        for stat_col, title, ax in stats_config:
            bars = ax.bar(country_stats.index, country_stats[stat_col], 
                         color=colors, alpha=0.8)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel(title.split("Average ")[1])
            
            # Add value labels on bars
            for bar, value in zip(bars, country_stats[stat_col]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:,.0f}', ha='center', va='bottom', fontsize=10)
            
            # Format y-axis for better readability
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self) -> None:
        """Generate a comprehensive summary report of the data."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        print("=== YOUTUBE DATA ANALYSIS SUMMARY ===\n")
        
        # Overall statistics
        print(f"Total videos analyzed: {len(self.df):,}")
        print(f"Countries: {', '.join([COUNTRIES[c] for c in self.df['country'].unique()])}")
        print(f"Date range: {self.df['publishedAt'].min()} to {self.df['publishedAt'].max()}")
        print()
        
        # Country breakdown
        print("Videos by Country:")
        country_counts = self.df['country'].value_counts()
        for country_code, count in country_counts.items():
            print(f"  {COUNTRIES.get(country_code, country_code)}: {count:,}")
        print()
        
        # Top categories overall
        print("Top 5 Categories (by total views):")
        top_categories_overall = (self.df.groupby("categoryName")["viewCount"]
                                 .sum().sort_values(ascending=False).head(5))
        for category, views in top_categories_overall.items():
            print(f"  {category}: {views:,} views")
        print()
        
        # Data quality info
        missing_data = self.df.isnull().sum()
        if missing_data.any():
            print("Missing Data Summary:")
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    pct = (missing_count / len(self.df)) * 100
                    print(f"  {col}: {missing_count:,} ({pct:.1f}%)")


def main():
    """Main execution function."""
    try:
        # Initialize analyzer
        analyzer = YouTubeAnalyzer()
        
        # Load and process data
        logger.info("Starting YouTube data analysis...")
        df = analyzer.load_all_data()
        
        if df.empty:
            logger.error("No data available for analysis")
            return
        
        # Generate summary report
        analyzer.generate_summary_report()
        
        # Create visualizations
        logger.info("Creating visualizations...")
        analyzer.plot_category_analysis(top_n=10)
        analyzer.plot_country_statistics()
        
        logger.info("Analysis complete!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()