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
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Streamlit page configuration
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COUNTRIES = {
    "FR": "France",
    "US": "United States", 
    "IN": "India"
}

# CPM rates by country (in EUR per 1000 views)
CPM_RATES = {
    "US": 10.263,
    "FR": 3.903,
    "IN": 0.826
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
                st.error(f"File not found: {path}")
                return {}
            
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {path}: {e}")
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
    
    def calculate_revenue(self, views: int, country: str) -> float:
        """Calculate estimated revenue based on views and country CPM."""
        if pd.isna(views) or views <= 0:
            return 0.0
        
        cpm = CPM_RATES.get(country, 0)
        # Revenue = (Views / 1000) * CPM
        return (views / 1000) * cpm
    
    def extract_video_data(self, data: Dict[str, Any], country: str) -> List[Dict[str, Any]]:
        """Extract video data from JSON and convert to list of dictionaries."""
        rows = []
        items = data.get("items", [])
        
        if not items:
            st.warning(f"No items found for country: {country}")
        
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
                view_count = self.safe_int_conversion(statistics.get("viewCount"))
                
                # Calculate estimated revenue
                estimated_revenue = self.calculate_revenue(view_count, country)
                
                row = {
                    "country": country,
                    "videoId": item.get("id"),
                    "title": snippet.get("title"),
                    "channelTitle": snippet.get("channelTitle"),
                    "categoryId": str(snippet.get("categoryId", "")),
                    "publishedAt": published_at,
                    "duration_s": duration_s,
                    "viewCount": view_count,
                    "likeCount": self.safe_int_conversion(statistics.get("likeCount")),
                    "commentCount": self.safe_int_conversion(statistics.get("commentCount")),
                    "tags_count": len(tags),
                    "cpm": CPM_RATES.get(country, 0),
                    "estimated_revenue": estimated_revenue
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
    
    @st.cache_data
    def load_all_data(_self) -> pd.DataFrame:
        """Load and process all YouTube data files."""
        all_rows = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_countries = len(DATA_FILES)
        
        for idx, (country_code, (data_file, categories_file)) in enumerate(DATA_FILES.items()):
            status_text.text(f"Processing data for {COUNTRIES[country_code]}...")
            progress_bar.progress((idx + 1) / total_countries)
            
            # Load video data
            video_data = _self.load_json(data_file)
            if not video_data:
                continue
            
            # Load category data
            categories_data = _self.load_json(categories_file)
            _self.category_maps[country_code] = _self.load_category_mapping(categories_data)
            
            # Extract video rows
            country_rows = _self.extract_video_data(video_data, country_code)
            all_rows.extend(country_rows)
        
        # Create DataFrame
        df = pd.DataFrame(all_rows)
        
        if df.empty:
            st.error("No data could be loaded successfully")
            return df
        
        # Data cleaning and preprocessing
        df = _self.clean_data(df)
        
        # Add category names
        df["categoryName"] = df.apply(_self.map_category_name, axis=1)
        
        status_text.text("Loading complete!")
        progress_bar.empty()
        status_text.empty()
        
        _self.df = df
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the DataFrame."""
        # Fill missing view counts with 0
        df["viewCount"] = df["viewCount"].fillna(0)
        
        # Convert to appropriate data types
        numeric_columns = ["viewCount", "likeCount", "commentCount", "duration_s", "tags_count", "estimated_revenue"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing essential data
        df = df.dropna(subset=["videoId", "country"])
        
        return df
    
    def map_category_name(self, row: pd.Series) -> str:
        """Map category ID to category name for a given row."""
        country = row["country"]
        category_id = row["categoryId"]
        
        category_map = self.category_maps.get(country, {})
        return category_map.get(category_id, "Unknown")
    
    def get_cpm_analysis_by_category(self) -> pd.DataFrame:
        """Calculate CPM analysis by category and country."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        # Group by country and category
        cpm_analysis = self.df.groupby(['country', 'categoryName']).agg({
            'viewCount': ['sum', 'mean', 'count'],
            'estimated_revenue': ['sum', 'mean'],
            'cpm': 'first'
        }).round(2)
        
        # Flatten column names
        cpm_analysis.columns = ['total_views', 'avg_views', 'video_count', 'total_revenue', 'avg_revenue', 'cpm']
        cpm_analysis = cpm_analysis.reset_index()
        
        # Add country full names
        cpm_analysis['country_name'] = cpm_analysis['country'].map(COUNTRIES)
        
        return cpm_analysis
    
    def plot_cpm_by_category_plotly(self):
        """Create interactive visualization of CPM and revenue by category."""
        cpm_data = self.get_cpm_analysis_by_category()
        
        # Create subplots with increased spacing for legend
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "CPM by Category and Country",
                "Average Revenue per Video",
                "Total Revenue by Category",
                "Total Views by Category"
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.15,  # Increased from 0.12
            horizontal_spacing=0.12  # Increased from 0.1
        )
        
        colors = px.colors.qualitative.Set2
        country_colors = {country: colors[i] for i, country in enumerate(COUNTRIES.keys())}
        
        # 1. CPM by category and country
        for country in COUNTRIES.keys():
            country_data = cpm_data[cpm_data['country'] == country]
            if not country_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=country_data['categoryName'],
                        y=country_data['cpm'],
                        name=f"{COUNTRIES[country]} - CPM",
                        marker_color=country_colors[country],
                        text=[f"€{v:.2f}" for v in country_data['cpm']],
                        textposition='outside',
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # 2. Average revenue per video
        for country in COUNTRIES.keys():
            country_data = cpm_data[cpm_data['country'] == country]
            if not country_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=country_data['categoryName'],
                        y=country_data['avg_revenue'],
                        name=f"{COUNTRIES[country]} - Rev/Video",
                        marker_color=country_colors[country],
                        text=[f"€{v:.0f}" for v in country_data['avg_revenue']],
                        textposition='outside',
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. Total revenue by category
        total_revenue_by_category = cpm_data.groupby('categoryName')['total_revenue'].sum().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(
                x=total_revenue_by_category.index,
                y=total_revenue_by_category.values,
                marker_color=colors[0],
                text=[f"€{v:,.0f}" for v in total_revenue_by_category.values],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Total views by category
        total_views_by_category = cpm_data.groupby('categoryName')['total_views'].sum().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(
                x=total_views_by_category.index,
                y=total_views_by_category.values,
                marker_color=colors[1],
                text=[f"{v:,.0f}" for v in total_views_by_category.values],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout with more space for legend
        fig.update_layout(
            title="CPM and Revenue Analysis by Category",
            height=850,  # Increased from 800
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,  # Position legend below the plots
                xanchor="center",
                x=0.5
            ),
            margin=dict(b=100)  # Add bottom margin for legend
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def plot_revenue_comparison_plotly(self):
        """Create revenue comparison visualization."""
        cpm_data = self.get_cpm_analysis_by_category()
        
        # Create a heatmap of average revenue by category and country
        pivot_data = cpm_data.pivot(index='categoryName', columns='country_name', values='avg_revenue')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            text=[[f"€{val:.0f}" if not pd.isna(val) else "N/A" for val in row] for row in pivot_data.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Heatmap of Average Revenue per Video (EUR)",
            xaxis_title="Country",
            yaxis_title="Category",
            height=650,  # Increased height
            margin=dict(b=80)  # Add bottom margin
        )
        
        return fig
    
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
    
    def plot_category_analysis_plotly(self, top_n: int = 10):
        """Create interactive bar plots showing top categories by view count for each country."""
        top_categories = self.get_top_categories_by_country(top_n)
        
        if not top_categories:
            st.warning("No category data available")
            return
        
        # Create subplot figure with more spacing
        fig = make_subplots(
            rows=1, cols=len(top_categories),
            subplot_titles=[COUNTRIES[code] for code in top_categories.keys()],
            horizontal_spacing=0.12  # Increased spacing
        )
        
        colors = px.colors.qualitative.Set2
        
        for idx, (country_code, categories) in enumerate(top_categories.items()):
            fig.add_trace(
                go.Bar(
                    y=categories.index,
                    x=categories.values,
                    orientation='h',
                    name=COUNTRIES[country_code],
                    marker_color=colors[idx % len(colors)],
                    text=[f'{v:,.0f}' for v in categories.values],
                    textposition='outside',
                    showlegend=False
                ),
                row=1, col=idx + 1
            )
            
            # Reverse y-axis to show highest at top
            fig.update_yaxes(autorange="reversed", row=1, col=idx + 1)
        
        fig.update_layout(
            title="Top Categories by Total View Count",
            height=650,  # Increased height
            showlegend=False,
            margin=dict(b=80)  # Add bottom margin
        )
        
        return fig
    
    def plot_country_statistics_plotly(self):
        """Create interactive bar plots showing average statistics by country."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        
        # Calculate statistics by country
        country_stats = self.df.groupby("country").agg({
            "viewCount": "mean",
            "likeCount": "mean", 
            "commentCount": "mean",
            "duration_s": "mean",
            "estimated_revenue": "mean"
        }).round(0)
        
        # Map country codes to full names
        country_stats.index = [COUNTRIES.get(code, code) for code in country_stats.index]
        
        # Create subplots with increased spacing
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                "Average Views",
                "Average Likes", 
                "Average Comments",
                "Average Duration (seconds)",
                "Average Revenue per Video (EUR)",
                "CPM by Country"
            ],
            vertical_spacing=0.15,  # Increased spacing
            horizontal_spacing=0.12  # Increased spacing
        )
        
        stats_config = [
            ("viewCount", 1, 1),
            ("likeCount", 1, 2),
            ("commentCount", 1, 3),
            ("duration_s", 2, 1),
            ("estimated_revenue", 2, 2)
        ]
        
        colors = px.colors.qualitative.Set2
        
        for stat_col, row, col in stats_config:
            fig.add_trace(
                go.Bar(
                    x=country_stats.index,
                    y=country_stats[stat_col],
                    marker_color=colors,
                    text=[f'{v:,.0f}' for v in country_stats[stat_col]],
                    textposition='outside',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Add CPM comparison
        cpm_values = [CPM_RATES[code] for code in COUNTRIES.keys()]
        fig.add_trace(
            go.Bar(
                x=list(COUNTRIES.values()),
                y=cpm_values,
                marker_color=colors,
                text=[f'€{v:.2f}' for v in cpm_values],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title="Average Video Statistics by Country (including CPM)",
            height=850,  # Increased height
            showlegend=False,
            margin=dict(b=80)  # Add bottom margin
        )
        
        return fig


def main():
    """Main Streamlit interface."""
    
    # Title and description
    st.title("📊 YouTube Analytics Dashboard with CPM")
    st.markdown("---")
    
    # Sidebar for controls
    st.sidebar.title("Settings")
    
    # Display CPM rates
    st.sidebar.subheader("CPM Rates (EUR per 1000 views)")
    for country_code, cpm in CPM_RATES.items():
        st.sidebar.write(f"🇺🇸 {COUNTRIES[country_code]}: €{cpm:.3f}" if country_code == "US" else 
                        f"🇫🇷 {COUNTRIES[country_code]}: €{cpm:.3f}" if country_code == "FR" else
                        f"🇮🇳 {COUNTRIES[country_code]}: €{cpm:.3f}")
    
    # Initialize analyzer
    analyzer = YouTubeAnalyzer()
    
    # Data loading
    with st.spinner("Loading data..."):
        df = analyzer.load_all_data()
    
    if df.empty:
        st.error("❌ No data available for analysis")
        st.stop()
    
    # Sidebar controls
    st.sidebar.subheader("Filters")
    
    # Country filter
    selected_countries = st.sidebar.multiselect(
        "Select countries:",
        options=list(COUNTRIES.keys()),
        default=list(COUNTRIES.keys()),
        format_func=lambda x: COUNTRIES[x]
    )
    
    # Filter dataframe
    if selected_countries:
        filtered_df = df[df['country'].isin(selected_countries)]
    else:
        filtered_df = df
    
    # Top N categories slider
    top_n = st.sidebar.slider("Number of categories to display:", 5, 15, 10)
    
    # Main dashboard
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Videos", f"{len(filtered_df):,}")
    
    with col2:
        avg_views = filtered_df['viewCount'].mean()
        st.metric("Average Views", f"{avg_views:,.0f}")
    
    with col3:
        avg_likes = filtered_df['likeCount'].mean()
        st.metric("Average Likes", f"{avg_likes:,.0f}")
    
    with col4:
        avg_duration = filtered_df['duration_s'].mean() / 60  # Convert to minutes
        st.metric("Average Duration", f"{avg_duration:.1f} min")
    
    with col5:
        total_revenue = filtered_df['estimated_revenue'].sum()
        st.metric("Total Estimated Revenue", f"€{total_revenue:,.0f}")
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💰 CPM & Revenue", "📈 Categories", "🌍 Countries", "📊 Data", "📋 Summary", "🔥 Heatmap"])
    
    with tab1:
        st.subheader("CPM and Revenue Analysis by Category")
        
        # Update analyzer's df with filtered data
        analyzer.df = filtered_df
        
        if len(selected_countries) > 0:
            fig_cpm = analyzer.plot_cpm_by_category_plotly()
            st.plotly_chart(fig_cpm, use_container_width=True)
            
            # Table with detailed CPM analysis
            st.subheader("Detailed CPM Analysis by Category")
            cpm_analysis = analyzer.get_cpm_analysis_by_category()
            
            # Format the dataframe for display
            display_df = cpm_analysis.copy()
            display_df['cpm'] = display_df['cpm'].apply(lambda x: f"€{x:.3f}")
            display_df['avg_revenue'] = display_df['avg_revenue'].apply(lambda x: f"€{x:.2f}")
            display_df['total_revenue'] = display_df['total_revenue'].apply(lambda x: f"€{x:,.0f}")
            display_df['total_views'] = display_df['total_views'].apply(lambda x: f"{x:,.0f}")
            display_df['avg_views'] = display_df['avg_views'].apply(lambda x: f"{x:,.0f}")
            
            display_df.columns = ['Country', 'Category', 'Total Views', 'Avg Views', 'Video Count', 
                                'Total Revenue', 'Avg Revenue/Video', 'CPM', 'Country Name']
            
            st.dataframe(
                display_df[['Country Name', 'Category', 'Video Count', 'Total Views', 'Avg Views', 
                           'CPM', 'Avg Revenue/Video', 'Total Revenue']],
                use_container_width=True
            )
        else:
            st.info("Please select at least one country.")
    
    with tab2:
        st.subheader("Analysis by Categories")
        
        # Update analyzer's df with filtered data
        analyzer.df = filtered_df
        
        if len(selected_countries) > 0:
            fig_categories = analyzer.plot_category_analysis_plotly(top_n)
            st.plotly_chart(fig_categories, use_container_width=True)
        else:
            st.info("Please select at least one country.")
    
    with tab3:
        st.subheader("Comparison by Country")
        
        if len(selected_countries) > 1:
            fig_countries = analyzer.plot_country_statistics_plotly()
            st.plotly_chart(fig_countries, use_container_width=True)
        else:
            st.info("Please select at least two countries for comparison.")
    
    with tab4:
        st.subheader("Data Exploration")
        
        # Category distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Distribution by Country**")
            country_counts = filtered_df['country'].value_counts()
            country_names = [COUNTRIES.get(code, code) for code in country_counts.index]
            
            fig_pie = px.pie(
                values=country_counts.values,
                names=country_names,
                title="Video Distribution by Country"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.write("**Top 10 Channels by Revenue**")
            top_channels_revenue = filtered_df.groupby('channelTitle')['estimated_revenue'].sum().nlargest(10)
            
            fig_channels = px.bar(
                x=top_channels_revenue.values,
                y=top_channels_revenue.index,
                orientation='h',
                title="Channels with Highest Estimated Revenue",
                labels={'x': 'Estimated Revenue (EUR)', 'y': 'Channel'}
            )
            fig_channels.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_channels, use_container_width=True)
        
        # Data table
        st.subheader("Data Table")
        st.write(f"Showing {len(filtered_df)} videos")
        
        # Select columns to display
        display_cols = st.multiselect(
            "Columns to display:",
            options=filtered_df.columns.tolist(),
            default=['title', 'channelTitle', 'categoryName', 'viewCount', 'likeCount', 'estimated_revenue', 'cpm']
        )
        
        if display_cols:
            # Format display dataframe
            display_data = filtered_df[display_cols].head(100).copy()
            if 'estimated_revenue' in display_cols:
                display_data['estimated_revenue'] = display_data['estimated_revenue'].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")
            if 'cpm' in display_cols:
                display_data['cpm'] = display_data['cpm'].apply(lambda x: f"€{x:.3f}" if pd.notna(x) else "N/A")
                
            st.dataframe(display_data, use_container_width=True)
    
    with tab5:
        st.subheader("Analysis Summary")
        
        # Summary statistics
        st.write("### General Statistics")
        
        summary_data = {
            "Metric": [
                "Total number of videos",
                "Countries analyzed",
                "Total estimated revenue",
                "Average revenue per video",
                "Date range",
                "Average number of views",
                "Average number of likes",
                "Average number of comments",
                "Average duration (minutes)"
            ],
            "Value": [
                f"{len(filtered_df):,}",
                ", ".join([COUNTRIES[c] for c in filtered_df['country'].unique()]),
                f"€{filtered_df['estimated_revenue'].sum():,.0f}",
                f"€{filtered_df['estimated_revenue'].mean():.2f}",
                f"{filtered_df['publishedAt'].min().strftime('%Y-%m-%d')} to {filtered_df['publishedAt'].max().strftime('%Y-%m-%d')}",
                f"{filtered_df['viewCount'].mean():,.0f}",
                f"{filtered_df['likeCount'].mean():,.0f}",
                f"{filtered_df['commentCount'].mean():,.0f}",
                f"{filtered_df['duration_s'].mean() / 60:.1f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
        
        # Top categories by revenue
        st.write("### Top 10 Categories (by total revenue)")
        top_categories_revenue = (filtered_df.groupby("categoryName")["estimated_revenue"]
                                 .sum().sort_values(ascending=False).head(10))
        
        revenue_data = {
            "Category": top_categories_revenue.index,
            "Total Estimated Revenue": [f"€{v:,.0f}" for v in top_categories_revenue.values]
        }
        
        revenue_df = pd.DataFrame(revenue_data)
        st.table(revenue_df)
        
        # Top categories overall by views
        st.write("### Top 10 Categories (by total views)")
        top_categories_overall = (filtered_df.groupby("categoryName")["viewCount"]
                                 .sum().sort_values(ascending=False).head(10))
        
        categories_data = {
            "Category": top_categories_overall.index,
            "Total Views": [f"{v:,}" for v in top_categories_overall.values]
        }
        
        categories_df = pd.DataFrame(categories_data)
        st.table(categories_df)
        
        # Data quality
        st.write("### Data Quality")
        missing_data = filtered_df.isnull().sum()
        if missing_data.any():
            quality_data = []
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    pct = (missing_count / len(filtered_df)) * 100
                    quality_data.append({
                        "Column": col,
                        "Missing Values": f"{missing_count:,}",
                        "Percentage": f"{pct:.1f}%"
                    })
            
            if quality_data:
                quality_df = pd.DataFrame(quality_data)
                st.table(quality_df)
            else:
                st.success("✅ No missing values detected!")
        else:
            st.success("✅ No missing values detected!")
    
    with tab6:
        st.subheader("Revenue Heatmap by Category and Country")
        
        # Update analyzer's df with filtered data
        analyzer.df = filtered_df
        
        if len(selected_countries) > 0:
            fig_heatmap = analyzer.plot_revenue_comparison_plotly()
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Additional analysis
            st.write("### Comparative CPM Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Revenue per Million Views**")
                revenue_per_million = {}
                for country_code in selected_countries:
                    country_data = filtered_df[filtered_df['country'] == country_code]
                    if not country_data.empty:
                        total_views = country_data['viewCount'].sum()
                        total_revenue = country_data['estimated_revenue'].sum()
                        rpm = (total_revenue / total_views) * 1000000 if total_views > 0 else 0
                        revenue_per_million[COUNTRIES[country_code]] = rpm
                
                rpm_df = pd.DataFrame(list(revenue_per_million.items()), 
                                    columns=['Country', 'Revenue per Million Views (EUR)'])
                rpm_df['Revenue per Million Views (EUR)'] = rpm_df['Revenue per Million Views (EUR)'].apply(lambda x: f"€{x:,.0f}")
                st.table(rpm_df)
            
            with col2:
                st.write("**Top 5 Most Profitable Categories**")
                # Calculate average revenue per view by category across all countries
                category_efficiency = (filtered_df.groupby('categoryName')
                                     .apply(lambda x: x['estimated_revenue'].sum() / x['viewCount'].sum() if x['viewCount'].sum() > 0 else 0)
                                     .sort_values(ascending=False)
                                     .head(5))
                
                efficiency_data = {
                    "Category": category_efficiency.index,
                    "Revenue per View (EUR)": [f"€{v:.6f}" for v in category_efficiency.values]
                }
                
                efficiency_df = pd.DataFrame(efficiency_data)
                st.table(efficiency_df)
                
            # Performance insights
            st.write("### Performance Insights")
            
            insights = []
            
            # Best performing country by average revenue per video
            avg_revenue_by_country = filtered_df.groupby('country')['estimated_revenue'].mean()
            best_country = avg_revenue_by_country.idxmax()
            best_revenue = avg_revenue_by_country.max()
            insights.append(f"🏆 **Most profitable country**: {COUNTRIES[best_country]} with €{best_revenue:.2f} average revenue per video")
            
            # Most popular category overall
            most_popular_category = filtered_df['categoryName'].value_counts().index[0]
            category_count = filtered_df['categoryName'].value_counts().iloc[0]
            insights.append(f"📊 **Most popular category**: {most_popular_category} with {category_count} videos")
            
            # Highest revenue generating category
            highest_revenue_category = filtered_df.groupby('categoryName')['estimated_revenue'].sum().idxmax()
            highest_revenue_amount = filtered_df.groupby('categoryName')['estimated_revenue'].sum().max()
            insights.append(f"💰 **Most profitable category**: {highest_revenue_category} with €{highest_revenue_amount:,.0f} total revenue")
            
            # CPM comparison insight
            cpm_values = [(COUNTRIES[k], v) for k, v in CPM_RATES.items()]
            cpm_values.sort(key=lambda x: x[1], reverse=True)
            insights.append(f"📈 **CPM difference**: {cpm_values[0][0]} (€{cpm_values[0][1]:.3f}) generates {cpm_values[0][1]/cpm_values[-1][1]:.1f}x more revenue per view than {cpm_values[-1][0]} (€{cpm_values[-1][1]:.3f})")
            
            for insight in insights:
                st.write(insight)
        else:
            st.info("Please select at least one country.")


if __name__ == "__main__":
    main()