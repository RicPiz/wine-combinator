"""
Wine Combinator – Streamlit app (Improved Version)

This app recommends wines from two CSV datasets (red and white) based on
user-selected filters (color, price, food pairing). It renders a responsive
grid of wine cards with bottle images, summary badges, a price line, and a
Details section that includes aromas and structure metrics.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import html as html_escape
import unicodedata

import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class AppConfig:
    """Application configuration constants."""
    TITLE: str = "Wine Combinator"
    DATA_FILES: Dict[str, str] = None
    IMAGES_DIR: str = "img_all"
    SIDEBAR_LOGOS: Dict[str, str] = None
    MAIN_LOGOS: Dict[str, str] = None
    IMAGE_SIZE: Tuple[int, int] = (260, 360)
    DEFAULT_TOP_K: int = 3
    GRID_COLS: int = 3
    CSV_ENCODING: str = 'iso8859_2'
    CSV_SEPARATOR: str = ';'
    
    def __post_init__(self):
        if self.DATA_FILES is None:
            self.DATA_FILES = {
                "Red": "red_wines_clean.csv",
                "White": "white_wines_clean.csv"
            }
        if self.SIDEBAR_LOGOS is None:
            self.SIDEBAR_LOGOS = {
                "svg": "Wlogo.svg",
                "png": "Wlogo.png"
            }
        if self.MAIN_LOGOS is None:
            self.MAIN_LOGOS = {
                "svg": "winecombinator.svg",
                "png": "winecombinator.png"
            }


config = AppConfig()


# -----------------------------
# Text Processing Utilities
# -----------------------------

def normalize_text(value: str) -> str:
    """Normalize text for filename/key matching."""
    if not isinstance(value, str):
        value = str(value)
    
    # Strip quotes and normalize
    value = value.strip().strip("'\"")
    value = unicodedata.normalize("NFKD", value)
    value = "".join([c for c in value if not unicodedata.combining(c)])
    
    # Clean and normalize
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    
    return value


def parse_food_tokens(food_advice: str) -> List[str]:
    """Parse the CSV food_advice field into high-level tokens."""
    if not isinstance(food_advice, str):
        return []
    
    # Split by slash and clean
    tokens = []
    for part in food_advice.split("/"):
        part = re.sub(r"\(.*?\)", "", part).strip()  # Remove parenthetical info
        if part and part.lower() not in {"na", "n.a.", "n/a", "none"}:
            tokens.append(part)
    
    return tokens


# -----------------------------
# Data Loading and Processing
# -----------------------------

class WineDataLoader:
    """Handles wine data loading and processing."""
    
    NUMERIC_COLS = ["avg_review", "num_review", "price", "year", 
                    "light_struct", "dry_sweet", "flat_acidic"]
    
    EXPECTED_COLS = [
        "company", "name", "avg_review", "num_review", "price", 
        "food_advice", "country", "region", "wine_type", "year",
        "light_struct", "dry_sweet", "flat_acidic", "notes"
    ]
    
    @classmethod
    @st.cache_data(show_spinner=False)
    def load_dataset(cls, kind: str) -> pd.DataFrame:
        """Load and clean a single dataset."""
        file_path = config.DATA_FILES[kind]
        df = pd.read_csv(
            file_path, 
            sep=config.CSV_SEPARATOR, 
            encoding=config.CSV_ENCODING
        )
        
        # Convert numeric columns
        for col in cls.NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Ensure all expected columns exist
        df = df.reindex(columns=cls.EXPECTED_COLS, fill_value=np.nan)
        
        # Add computed columns
        df["wine_color"] = kind
        df["food_tokens"] = df["food_advice"].fillna("").apply(parse_food_tokens)
        df["image_key_candidates"] = df.apply(cls._make_image_keys, axis=1)
        
        return df
    
    @staticmethod
    def _make_image_keys(row: pd.Series) -> List[str]:
        """Generate candidate keys for image matching."""
        name = str(row.get("name", "")).strip()
        year = row.get("year")
        company = str(row.get("company", "")).strip()
        
        candidates = []
        
        # Try different combinations
        if name and not pd.isna(year):
            candidates.append(normalize_text(f"{name} {int(year)}"))
        if name:
            candidates.append(normalize_text(name))
        if company and name:
            if not pd.isna(year):
                candidates.append(normalize_text(f"{company} {name} {int(year)}"))
            candidates.append(normalize_text(f"{company} {name}"))
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(candidates))
    
    @classmethod
    @st.cache_data(show_spinner=False)
    def load_all_data(cls) -> pd.DataFrame:
        """Load both datasets and concatenate."""
        frames = [cls.load_dataset(color) for color in ["Red", "White"]]
        return pd.concat(frames, ignore_index=True)


# -----------------------------
# Image Management
# -----------------------------

class ImageManager:
    """Handles image indexing and loading."""
    
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def index_images(images_dir: str) -> Dict[str, str]:
        """Build index mapping normalized filenames to paths."""
        mapping = {}
        if not os.path.isdir(images_dir):
            return mapping
        
        for file_path in Path(images_dir).glob("*.png"):
            stem = file_path.stem.rstrip("'")
            key = normalize_text(stem)
            mapping[key] = str(file_path)
        
        return mapping
    
    @staticmethod
    def find_image(index: Dict[str, str], candidates: List[str]) -> Optional[str]:
        """Find the first matching image path."""
        # Try exact matches
        for key in candidates:
            if key in index:
                return index[key]
        
        # Fallback: substring search
        for key in candidates:
            if key:
                for indexed_key, path in index.items():
                    if key in indexed_key:
                        return path
        
        return None
    
    @staticmethod
    @st.cache_data(show_spinner=False)
    def load_thumbnail(image_path: str, size: Tuple[int, int] = None) -> Optional[Image.Image]:
        """Load and resize an image to thumbnail size."""
        if size is None:
            size = config.IMAGE_SIZE
        
        try:
            with Image.open(image_path) as img:
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA")
                
                fill_color = (255, 255, 255, 0) if img.mode == "RGBA" else (255, 255, 255)
                padded = ImageOps.pad(
                    img, size, 
                    method=Image.LANCZOS, 
                    color=fill_color, 
                    centering=(0.5, 0.5)
                )
                return padded
        except Exception:
            return None


# -----------------------------
# Wine Filtering and Scoring
# -----------------------------

class WineFilter:
    """Handles wine filtering and scoring logic."""
    
    @staticmethod
    def compute_score(df: pd.DataFrame, price_range: Tuple[float, float]) -> pd.Series:
        """Score wines for ranking based on quality, popularity, and price fit."""
        avg = df["avg_review"].fillna(0.0).clip(lower=0.0)
        reviews = df["num_review"].fillna(0.0).clip(lower=0.0)
        price = df["price"].fillna(np.nan)
        
        # Normalize components
        avg_norm = (avg - avg.min()) / (avg.max() - avg.min() + 1e-9)
        
        reviews_norm = np.log1p(reviews)
        reviews_norm = (reviews_norm - reviews_norm.min()) / (reviews_norm.max() - reviews_norm.min() + 1e-9)
        
        # Price proximity
        center = (price_range[0] + price_range[1]) / 2.0
        price_dist = (price - center).abs()
        price_prox = 1.0 - (price_dist - price_dist.min()) / (price_dist.max() - price_dist.min() + 1e-9)
        price_prox = price_prox.fillna(0.0)
        
        # Weighted score
        return 0.6 * avg_norm + 0.3 * reviews_norm + 0.1 * price_prox
    
    @staticmethod
    def filter_by_food(df: pd.DataFrame, selected_foods: List[str]) -> pd.DataFrame:
        """Filter wines matching selected food tokens."""
        if not selected_foods:
            return df
        
        mask = df["food_tokens"].apply(
            lambda toks: any(tok in selected_foods for tok in toks)
        )
        return df[mask]
    
    @staticmethod
    def get_food_options(df: pd.DataFrame) -> List[str]:
        """Get all unique food pairing options."""
        foods: Set[str] = set()
        for tokens in df["food_tokens"]:
            for tok in tokens:
                if tok:  # Only add non-empty tokens
                    foods.add(tok)
        
        # Prioritize common items
        priority = ["Beef", "Lamb", "Veal", "Pasta", "Game", 
                   "Poultry", "Pork", "Fish", "Cheese", "Vegetarian"]
        
        ordered = [f for f in priority if f in foods]
        ordered.extend(sorted(foods - set(priority)))
        
        return ordered


# -----------------------------
# UI Components
# -----------------------------

class UIComponents:
    """Handles UI rendering components."""
    
    @staticmethod
    def inject_custom_css():
        """Inject custom CSS for consistent styling."""
        css = """
        <style>
        .wc-card { display: flex; flex-direction: column; height: 100%; }
        .wc-card-text {
            flex: 1 1 auto;
            min-height: 220px;
            max-height: 220px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .wc-title {
            font-weight: 700; margin-top: .25rem;
            display: -webkit-box; -webkit-line-clamp: 2; 
            -webkit-box-orient: vertical; overflow: hidden;
        }
        .wc-caption {
            opacity: 0.8; margin-bottom: .25rem;
            display: -webkit-box; -webkit-line-clamp: 2; 
            -webkit-box-orient: vertical; overflow: hidden;
        }
        .wc-badges, .wc-pair { opacity: 0.9; }
        .wc-pair { 
            display: -webkit-box; -webkit-line-clamp: 2; 
            -webkit-box-orient: vertical; overflow: hidden; 
        }
        .wc-price { margin: .25rem 0; }
        .wc-label { color: #722F37; font-weight: 600; margin: 0.25rem 0 0.25rem; }
        .wc-card [data-testid="stExpander"] { margin-top: auto; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    @staticmethod
    def show_logo(logo_type: str = "sidebar"):
        """Display logo based on type."""
        if logo_type == "sidebar":
            logos = config.SIDEBAR_LOGOS
            width = 120
            container = st.sidebar
        else:
            logos = config.MAIN_LOGOS
            width = None
            container = st
        
        # Try SVG first, then PNG
        for ext in ["svg", "png"]:
            if Path(logos[ext]).exists():
                if logo_type == "sidebar":
                    col1, col2, col3 = container.columns([1, 1, 1])
                    with col2:
                        st.image(logos[ext], width=width)
                else:
                    cols = st.columns([1, 3, 1])
                    with cols[1]:
                        st.image(logos[ext], use_container_width=True)
                break
    
    @staticmethod
    def render_sidebar_filters(df: pd.DataFrame) -> Dict:
        """Render sidebar filters and return selections."""
        st.sidebar.subheader("Filters")
        
        # Wine color
        color_choice = st.sidebar.multiselect(
            "Wine color",
            options=["Red", "White"],
            default=["Red", "White"]
        )
        st.sidebar.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        
        # Apply color filter first
        df_filtered = df[df["wine_color"].isin(color_choice)]
        
        # Food pairing
        food_options = WineFilter.get_food_options(df_filtered)
        selected_foods = st.sidebar.multiselect(
            "Food pairing",
            options=food_options,
            help="Choose one or more food types to match."
        )
        st.sidebar.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        
        # Price range
        prices = df_filtered["price"].dropna()
        min_price = float(prices.min()) if not prices.empty else 0.0
        max_price = float(prices.max()) if not prices.empty else 1000.0
        
        price_range = st.sidebar.slider(
            "Price range (EUR)",
            min_value=float(max(0.0, np.floor(min_price))),
            max_value=float(np.ceil(max_price)),
            value=(float(np.floor(min_price)), float(np.ceil(max_price))),
            step=1.0
        )
        st.sidebar.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        
        # Additional options
        top_k = st.sidebar.number_input(
            "Number of recommendations", 
            min_value=1, 
            max_value=33, 
            value=config.DEFAULT_TOP_K, 
            step=1
        )
        sort_by_price = st.sidebar.toggle("Sort by price (ascending)", value=True)
        
        return {
            "colors": color_choice,
            "foods": selected_foods,
            "price_range": price_range,
            "top_k": top_k,
            "sort_by_price": sort_by_price
        }
    
    @staticmethod
    def render_wine_card(row: pd.Series, image_manager: ImageManager, image_index: Dict[str, str]):
        """Render a single wine card."""
        st.markdown("<div class='wc-card'>", unsafe_allow_html=True)
        
        # Display image
        img_path = image_manager.find_image(image_index, row["image_key_candidates"])
        if img_path and Path(img_path).exists():
            thumb = image_manager.load_thumbnail(img_path)
            if thumb:
                st.image(thumb, width=260)
            else:
                # Fallback to placeholder if loading fails
                placeholder_path = "img_all/wine_placeholder.png"
                placeholder = image_manager.load_thumbnail(placeholder_path)
                if placeholder:
                    st.image(placeholder, width=260)
                else:
                    st.write(":wine_glass: Image not available")
        else:
            # Use placeholder for missing images
            placeholder_path = "img_all/wine_placeholder.png"
            placeholder = image_manager.load_thumbnail(placeholder_path)
            if placeholder:
                st.image(placeholder, width=260)
            else:
                st.write(":wine_glass: Image not available")
        
        # Prepare card content
        year = int(row['year']) if not pd.isna(row['year']) else 'NV'
        title = f"{row['name'].strip()} ({year})"
        
        badges = []
        if not pd.isna(row['avg_review']):
            badges.append(f"Rating: {row['avg_review']:.1f}")
        if not pd.isna(row['num_review']):
            badges.append(f"Reviews: {int(row['num_review'])}")
        badges.append(f"Color: {row['wine_color']}")
        
        pairing = ", ".join(row["food_tokens"]) if row["food_tokens"] else ""
        price = f"€{row['price']:,.2f}" if not pd.isna(row['price']) else "-"
        
        caption = f"{row['company']} • {row['wine_type']} • {row['region']}, {row['country']}"
        
        # Render HTML content
        html_block = f"""
        <div class='wc-card-text'>
          <div class='wc-title'>{html_escape.escape(title)}</div>
          <div class='wc-caption'>{html_escape.escape(caption)}</div>
          <div class='wc-badges'>{html_escape.escape(' • '.join(badges))}</div>
          <div class='wc-price'>Price: {html_escape.escape(price)}</div>
          <div class='wc-pair'>{('Pairing: ' + html_escape.escape(pairing)) if pairing else ''}</div>
        </div>
        """
        st.markdown(html_block, unsafe_allow_html=True)
        
        # Details expander
        with st.expander("Details"):
            notes = str(row.get("notes", "")).replace("/", ", ").strip()
            st.write(f"Aromas: {notes}" if notes else "Aromas: -")
            
            def fmt(v):
                if v is None or pd.isna(v):
                    return "-"
                try:
                    return f"{int(float(v))}/100"
                except:
                    return "-"
            
            st.markdown(
                "\n".join([
                    "Structure:",
                    f"- Light - Structured: {fmt(row.get('light_struct'))}",
                    f"- Dry - Sweet: {fmt(row.get('dry_sweet'))}",
                    f"- Flat - Acidic: {fmt(row.get('flat_acidic'))}",
                ])
            )
        
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Main Application
# -----------------------------

class WineCombinatorApp:
    """Main application class."""
    
    def __init__(self):
        self.data_loader = WineDataLoader()
        self.image_manager = ImageManager()
        self.ui = UIComponents()
        self.filter = WineFilter()
    
    def run(self):
        """Run the Streamlit application."""
        # Page configuration
        st.set_page_config(
            page_title=config.TITLE, 
            page_icon="🍷", 
            layout="wide"
        )
        
        # Display header and inject CSS
        self.ui.show_logo("main")
        cols = st.columns([1, 3, 1])
        with cols[1]:
            st.markdown(
                "<p style='text-align: center; font-size: 12px;'><i>Disclaimer: The data and images used in this work were obtained from the Vivino platform (https://www.vivino.com). Only information relating to Italian wines is included.</i></p>",
                unsafe_allow_html=True
            )
        self.ui.inject_custom_css()
        self.ui.show_logo("sidebar")
        
        # Load data
        df_all = self.data_loader.load_all_data()
        image_index = self.image_manager.index_images(config.IMAGES_DIR)

        view_mode = st.sidebar.radio("View Mode", ["Recommendations", "Wineries Map"])

        if view_mode == "Recommendations":
            # Get filter selections
            filters = self.ui.render_sidebar_filters(df_all)
            
            # Apply filters
            df_filtered = self._apply_filters(df_all, filters)
            
            # Display results
            self._display_results(df_filtered, image_index, filters)
        else:
            self._show_map_view(df_all)
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply all filters to the dataframe."""
        # Filter by color
        df = df[df["wine_color"].isin(filters["colors"])]
        
        # Filter by price
        df = df[
            (df["price"] >= filters["price_range"][0]) & 
            (df["price"] <= filters["price_range"][1])
        ]
        
        # Filter by food
        df = self.filter.filter_by_food(df, filters["foods"])
        
        if df.empty:
            return df
        
        # Add score and sort
        df = df.copy()
        df["rec_score"] = self.filter.compute_score(df, filters["price_range"])
        
        if filters["sort_by_price"]:
            df = df.sort_values(by=["price", "avg_review"], ascending=[True, False])
        else:
            df = df.sort_values(by=["rec_score", "avg_review"], ascending=[False, False])
        
        return df.head(filters["top_k"])
    
    def _display_results(self, df: pd.DataFrame, image_index: Dict[str, str], filters: Dict):
        """Display the recommendation results."""
        st.subheader("Recommendations")
        st.caption(
            "Based on your filters, we rank wines by rating, popularity, "
            "and closeness to your chosen price."
        )
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        
        if df.empty:
            st.info(
                "No wines match your current filters. Try widening the price range "
                "or removing some food pairings."
            )
            return
        
        # Display wine cards in grid
        cols = st.columns(config.GRID_COLS)
        for idx, (_, row) in enumerate(df.iterrows()):
            with cols[idx % config.GRID_COLS]:
                self.ui.render_wine_card(row, self.image_manager, image_index)
        
        # Download button
        csv_data = df.drop(columns=["image_key_candidates", "food_tokens"], errors="ignore")
        st.download_button(
            label="Download recommendations (CSV)",
            data=csv_data.to_csv(index=False),
            file_name="wine_recommendations.csv",
            mime="text/csv"
        )

    def _show_map_view(self, df_all: pd.DataFrame):
        st.subheader("Wineries Map")
        st.write("Click on a region to see the wineries and their wines.")

        @st.cache_data
        def load_map_data():
            regions = gpd.read_file('Wine_Map/Reg01012022_g/Reg01012022_g_WGS84.shp')
            regions = regions.to_crs('EPSG:4326')

            red_coords = pd.read_csv('Wine_Map/coordinate_red_wines_clean.csv', sep=';', encoding_errors='replace', encoding='iso8859_2')
            white_coords = pd.read_csv('Wine_Map/coordinate_white_wines_clean.csv', sep=';', encoding_errors='replace', encoding='iso8859_2')

            red_coords = red_coords.rename(columns={'winemaker': 'company', 'latitude': 'lat', 'longitude': 'long'})
            white_coords = white_coords.rename(columns={'winemaker': 'company', 'latitude': 'lat', 'longitude': 'long'})

            coords = pd.concat([white_coords, red_coords]).drop_duplicates(subset='company')

            coords['lat'] = coords['lat'].astype(float)
            coords['long'] = coords['long'].astype(float)
            coords = coords[(coords['long'] > 6) & (coords['long'] < 19) & (coords['lat'] > 35) & (coords['lat'] < 47)]

            gdf = gpd.GeoDataFrame(coords, geometry=gpd.points_from_xy(coords.long, coords.lat), crs='EPSG:4326')

            gdf = gpd.sjoin(gdf, regions, how='inner')

            # Add color based on wines
            gdf['marker_color'] = '#722F37'

            return regions, gdf

        regions, gdf = load_map_data()

        m = folium.Map(location=[41.9, 12.5], zoom_start=6, tiles='cartodbpositron')

        folium.GeoJson(
            regions,
            style_function=lambda feature: {
                'fillColor': 'lightgray',
                'color': 'gray',
                'weight': 1,
                'fillOpacity': 0.3
            },
            highlight_function=lambda feature: {
                'weight': 3,
                'fillOpacity': 0.5
            },
            tooltip=folium.GeoJsonTooltip(fields=['DEN_REG'], aliases=['Region:'], localize=True)
        ).add_to(m)

        for _, row in gdf.iterrows():
            folium.CircleMarker(
                location=(row['lat'], row['long']),
                radius=5,
                color=row['marker_color'],
                fill=True,
                fill_opacity=0.7,
                popup=row['company'],
                tooltip=row['company']
            ).add_to(m)

        map_data = st_folium(m, width=700, height=500, returned_objects=["last_clicked"])

        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lng = map_data["last_clicked"]["lng"]
            click_point = gpd.GeoDataFrame(geometry=[Point(click_lng, click_lat)], crs='EPSG:4326')

            joined = gpd.sjoin(click_point, regions, how='left', predicate='within')
            if not joined.empty:
                selected_region = joined['DEN_REG'].iloc[0]
                companies = gdf[gdf['DEN_REG'] == selected_region]['company'].unique()
                st.write(f"### Wines from {selected_region}")
                filtered_df = df_all[df_all['company'].isin(companies)][['company', 'name', 'year', 'wine_color', 'price']]
            else:
                st.write("No region selected.")
                return

            if not filtered_df.empty:
                display_df = filtered_df.rename(columns={
                    'company': 'Company',
                    'name': 'Name',
                    'year': 'Year',
                    'wine_color': 'Wine',
                    'price': 'Price'
                })[['Company', 'Name', 'Year', 'Wine', 'Price']]
                display_df['Price'] = display_df['Price'].apply(lambda x: f'€{x:.2f}' if pd.notna(x) else 'N/A')
                display_df = display_df.sort_values(by='Company')
                st.dataframe(display_df, hide_index=True)
            else:
                st.info("No wines found for this selection.")


def main():
    """Entry point."""
    app = WineCombinatorApp()
    app.run()


if __name__ == "__main__":
    main()
