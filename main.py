import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import isodate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="YouTube Analytics Dashboard - Live Data",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COUNTRIES = {
    "FR": "France",
    "US": "United States", 
    "IN": "India"
}

CPM_RATES = {
    "US": 10.263,
    "FR": 3.903,
    "IN": 0.826
}

file = Path("api_key.txt")

if not file.exists():
    st.error("Missing 'api_key.txt'. Please add your YouTube Data API v3 key. See 'README.md' for details. https://developers.google.com/youtube/v3/getting-started")
    st.stop()

API_KEY = file.read_text(encoding="utf-8").strip()

if not API_KEY:
    st.error("API key file is empty. Please ensure 'api_key.txt' contains your key.")
    st.stop()


BASE_URL = "https://www.googleapis.com/youtube/v3"

API_URLS = {
    "FR": f"{BASE_URL}/videos?part=snippet,contentDetails,statistics&chart=mostPopular&regionCode=FR&maxResults=50&key={API_KEY}",
    "IN": f"{BASE_URL}/videos?part=snippet,contentDetails,statistics&chart=mostPopular&regionCode=IN&maxResults=50&key={API_KEY}",
    "US": f"{BASE_URL}/videos?part=snippet,contentDetails,statistics&chart=mostPopular&regionCode=US&maxResults=50&key={API_KEY}"
}

CATEGORY_URLS = {
    "FR": f"{BASE_URL}/videoCategories?part=snippet&regionCode=FR&key={API_KEY}",
    "IN": f"{BASE_URL}/videoCategories?part=snippet&regionCode=IN&key={API_KEY}",
    "US": f"{BASE_URL}/videoCategories?part=snippet&regionCode=US&key={API_KEY}"
}

class YouTubeAnalyzer:
    """A class to analyze YouTube most popular videos data across countries using live API data."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.category_maps: Dict[str, Dict[str, str]] = {}
    
    def fetch_api_data(self, url: str) -> Dict[str, Any]:
        """Fetch data from YouTube API with error handling."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data from API: {e}")
            logger.error(f"API request failed: {e}")
            return {}
        except json.JSONDecodeError as e:
            st.error(f"Error parsing API response: {e}")
            logger.error(f"JSON decode error: {e}")
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
        """Extract video data from API response and convert to list of dictionaries."""
        rows = []
        items = data.get("items", [])
        
        if not items:
            st.warning(f"No items found for country: {country}")
        
        for item in items:
            try:
                snippet = item.get("snippet", {})
                statistics = item.get("statistics", {})
                content_details = item.get("contentDetails", {})
                
                published_at = self.parse_datetime(snippet.get("publishedAt"))
                duration_s = self.parse_duration(content_details.get("duration"))
                tags = snippet.get("tags") or []
                view_count = self.safe_int_conversion(statistics.get("viewCount"))
                
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
    
    def load_all_data(self) -> pd.DataFrame:
        """Load and process all YouTube data from live API."""
        all_rows = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_countries = len(API_URLS)
        
        for idx, (country_code, api_url) in enumerate(API_URLS.items()):
            status_text.text(f"Fetching live data for {COUNTRIES[country_code]}...")
            progress_bar.progress((idx + 0.5) / total_countries)
            
            video_data = self.fetch_api_data(api_url)
            if not video_data:
                st.warning(f"Failed to fetch video data for {COUNTRIES[country_code]}")
                continue
            
            category_url = CATEGORY_URLS.get(country_code)
            if category_url:
                categories_data = self.fetch_api_data(category_url)
                self.category_maps[country_code] = self.load_category_mapping(categories_data)
            
            country_rows = self.extract_video_data(video_data, country_code)
            all_rows.extend(country_rows)
            
            progress_bar.progress((idx + 1) / total_countries)
        
        df = pd.DataFrame(all_rows)
        
        if df.empty:
            st.error("No data could be loaded successfully from the APIs")
            return df
        
        df = self.clean_data(df)
    
        df["categoryName"] = df.apply(self.map_category_name, axis=1)
        
        status_text.text("Live data loading complete!")
        progress_bar.empty()
        
        st.success(f"Live data loaded successfully! Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        status_text.empty()
        
        self.df = df
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the DataFrame."""

        df["viewCount"] = df["viewCount"].fillna(0)
        
        numeric_columns = ["viewCount", "likeCount", "commentCount", "duration_s", "tags_count", "estimated_revenue"]
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
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
        
        cpm_analysis = self.df.groupby(['country', 'categoryName']).agg({
            'viewCount': ['sum', 'mean', 'count'],
            'estimated_revenue': ['sum', 'mean'],
            'cpm': 'first'
        }).round(2)
        
        cpm_analysis.columns = ['total_views', 'avg_views', 'video_count', 'total_revenue', 'avg_revenue', 'cpm']
        cpm_analysis = cpm_analysis.reset_index()
        
        cpm_analysis['country_name'] = cpm_analysis['country'].map(COUNTRIES)
        
        return cpm_analysis
    
    def plot_cpm_by_category_plotly(self):
        """Create interactive visualization of CPM and revenue by category."""
        cpm_data = self.get_cpm_analysis_by_category()
        
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
                        name=f"{COUNTRIES[country]}",
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
            height=850,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            margin=dict(b=100)
        )
        
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
            title="Heatmap of Average Revenue per Video - Live Data (EUR)",
            xaxis_title="Country",
            yaxis_title="Category",
            height=650,
            margin=dict(b=80)
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
        
        fig = make_subplots(
            rows=1, cols=len(top_categories),
            subplot_titles=[COUNTRIES[code] for code in top_categories.keys()],
            horizontal_spacing=0.12 
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
            
            fig.update_yaxes(autorange="reversed", row=1, col=idx + 1)
        
        fig.update_layout(
            title="Top Categories by Total View Count",
            height=650,
            showlegend=False,
            margin=dict(b=80)
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
            vertical_spacing=0.15,
            horizontal_spacing=0.12
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
            title="Average Video Statistics by Country",
            height=850, 
            showlegend=False,
            margin=dict(b=80)
        )
        
        return fig


def main():
    """Main Streamlit interface."""
    
    st.title("YouTube Analytics Dashboard")
    st.markdown("**Live Data** | Real-time analysis of YouTube's most popular videos")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(" **Data Source**: YouTube Data API v3")
    with col2:
        st.info(" **Regions**: France, India, United States")
    with col3:
        st.info(" **Videos per Country**: Up to 50 most popular")
    
    st.sidebar.title("Settings")
    
    # if st.sidebar.button("Refresh Data", help="Fetch latest data from YouTube API"):
    #     st.cache_data.clear()
    #     st.experimental_rerun()
    
    st.sidebar.subheader("CPM Rates (EUR per 1000 views)")
    for country_code, cpm in CPM_RATES.items():
        flag = "🇺🇸" if country_code == "US" else "🇫🇷" if country_code == "FR" else "🇮🇳"
        st.sidebar.write(f"{flag} {COUNTRIES[country_code]}: €{cpm:.3f}")
    
    analyzer = YouTubeAnalyzer()

    with st.spinner("Fetching live data from YouTube API..."):
        df = analyzer.load_all_data()
    
    if df.empty:
        st.error("No data available for analysis")
        st.info("This might be due to API quota limits or connectivity issues. Please try again later.")
        st.stop()
    
    st.sidebar.subheader("Filters")
    
    selected_countries = st.sidebar.multiselect(
        "Select countries:",
        options=list(COUNTRIES.keys()),
        default=list(COUNTRIES.keys()),
        format_func=lambda x: COUNTRIES[x]
    )
    
    if selected_countries:
        filtered_df = df[df['country'].isin(selected_countries)]
    else:
        filtered_df = df
    
    # top_n = st.sidebar.slider("Number of categories to display:", 5, 15, 10)
    top_n = 10
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["💰 CPM & Revenue", "📈 Categories", "🌍 Countries", "📊 Data"])
    
    with tab1:
        st.subheader("CPM and Revenue Analysis by Category")
    
        analyzer.df = filtered_df
        
        if len(selected_countries) > 0:
            fig_cpm = analyzer.plot_cpm_by_category_plotly()
            st.plotly_chart(fig_cpm, use_container_width=True)
            
            st.subheader("Detailed CPM Analysis by Category")
            cpm_analysis = analyzer.get_cpm_analysis_by_category()
            
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
        st.subheader("Live Data Exploration")
    
        st.write("**Top 10 Channels by Revenue**")
        top_channels_revenue = filtered_df.groupby('channelTitle')['estimated_revenue'].sum().nlargest(10)
        
        fig_channels = px.bar(
            x=top_channels_revenue.values,
            y=top_channels_revenue.index,
            orientation='h',
            title="Channels with Highest Estimated Revenue (Live)",
            labels={'x': 'Estimated Revenue (EUR)', 'y': 'Channel'}
        )
        fig_channels.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_channels, use_container_width=True)
        
        st.subheader("Live Data Table")
        st.write(f"Showing {len(filtered_df)} videos")
        
        display_cols = st.multiselect(
            "Columns to display:",
            options=filtered_df.columns.tolist(),
            default=['title', 'channelTitle', 'categoryName', 'viewCount', 'likeCount', 'estimated_revenue', 'cpm']
        )
        
        if display_cols:
            display_data = filtered_df[display_cols].head(100).copy()
            if 'estimated_revenue' in display_cols:
                display_data['estimated_revenue'] = display_data['estimated_revenue'].apply(lambda x: f"€{x:.2f}" if pd.notna(x) else "N/A")
            if 'cpm' in display_cols:
                display_data['cpm'] = display_data['cpm'].apply(lambda x: f"€{x:.3f}" if pd.notna(x) else "N/A")
                
            st.dataframe(display_data, use_container_width=True)

if __name__ == "__main__":
    main()