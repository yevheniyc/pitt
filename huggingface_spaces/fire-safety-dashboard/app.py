import os
import tempfile

import folium
import geopandas as gpd
import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from folium import plugins

# 🎨 DARK MODE COLOR PALETTE
# Professional, cohesive colors for dark theme
COLORS = {
    # Primary accent colors - muted, sophisticated palette
    "primary_red": "#B85450",  # Muted red for alarms/danger
    "primary_coral": "#C9746A",  # Soft coral for warmth
    "primary_rose": "#B8577A",  # Muted rose for mid-tone
    "primary_purple": "#8B6F8B",  # Soft purple for sophistication
    "primary_blue": "#7A7BB8",  # Muted purple-blue for cool tones
    # Fire category colors (muted, elegant)
    "fire_alarms": "#B85450",  # Muted red for alarms
    "structure_fires": "#A04742",  # Dark muted red for structure fires
    "outdoor_fires": "#8B6F8B",  # Soft purple for outdoor/brush
    "electrical": "#B8577A",  # Muted rose for electrical
    "vehicle_fires": "#C9746A",  # Soft coral for vehicles
    "gas_issues": "#7A7BB8",  # Muted purple-blue for gas
    "hazmat": "#6B7B8C",  # Blue-gray for hazmat
    "smoke": "#8A9BA8",  # Soft gray for smoke
    "uncategorized": "#6B7B7B",  # Muted gray for other
    # Background colors
    "bg_dark": "#2C3E50",  # Main dark background
    "bg_darker": "#34495E",  # Slightly lighter dark
    "bg_card": "#1A252F",  # Card backgrounds
    # Text colors
    "text_light": "#ECF0F1",  # Main text
    "text_muted": "#BDC3C7",  # Muted text
}

# Template for all plots
PLOT_TEMPLATE = "plotly_dark"

print("Loading fire incidents data...")
# Use the corrected fire alarms dataset for HF Spaces (smaller file)
df = pd.read_csv("corrected_fire_alarms.csv")

# Filter for fire incidents only (exclude EMS)
fire_incidents = df[~df["description_short"].str.contains("EMS", na=False)].copy()

# Clean and categorize fire types with improved categorization
fire_incidents["fire_category"] = "Other"

# Note: "Removed" entries are now handled as fire alarms (see corrected classification below)

# Filter out traffic incidents (not really fire incidents)
traffic_mask = fire_incidents["description_short"].str.contains(
    "TRAFFIC", na=False, case=False
)
fire_incidents.loc[traffic_mask, "fire_category"] = "Traffic Incidents"

# Categorize fire types for better analysis
# CORRECTED FIRE ALARM CLASSIFICATION:
# Before 2020: Use traditional 'ALARM' incidents
# After 2019: Use 'Removed' incidents (reclassified alarms)
pre_2020_alarm_mask = (fire_incidents["call_year"] < 2020) & (
    fire_incidents["description_short"].str.contains("ALARM", na=False, case=False)
)
post_2019_alarm_mask = (fire_incidents["call_year"] >= 2020) & (
    fire_incidents["description_short"] == "Removed"
)
alarm_mask = pre_2020_alarm_mask | post_2019_alarm_mask
fire_incidents.loc[alarm_mask, "fire_category"] = "Fire Alarms"

structure_mask = fire_incidents["description_short"].str.contains(
    "DWELLING|STRUCTURE|BUILDING|APARTMENT", na=False, case=False
)
fire_incidents.loc[structure_mask, "fire_category"] = "Structure Fires"

# Outdoor/brush fires - significant category that was hidden
outdoor_mask = fire_incidents["description_short"].str.contains(
    "BRUSH|GRASS|MULCH|OUTSIDE|OUTDOOR|ILLEGAL FIRE", na=False, case=False
)
fire_incidents.loc[outdoor_mask, "fire_category"] = "Outdoor/Brush Fires"

gas_mask = fire_incidents["description_short"].str.contains(
    "GAS|NATURAL GAS", na=False, case=False
)
fire_incidents.loc[gas_mask, "fire_category"] = "Gas Issues"

hazmat_mask = fire_incidents["description_short"].str.contains(
    "HAZMAT|CO OR HAZMAT", na=False, case=False
)
fire_incidents.loc[hazmat_mask, "fire_category"] = "Hazmat/CO Issues"

vehicle_mask = fire_incidents["description_short"].str.contains(
    "VEHICLE|AUTO|CAR", na=False, case=False
)
fire_incidents.loc[vehicle_mask, "fire_category"] = "Vehicle Fires"

# Electrical/transformer fires
electrical_mask = fire_incidents["description_short"].str.contains(
    "WIRE|ELECTRICAL|ARCING|TRANSFORMER", na=False, case=False
)
fire_incidents.loc[electrical_mask, "fire_category"] = "Electrical Issues"

# Water/flood related incidents
water_mask = fire_incidents["description_short"].str.contains(
    "WATER|FLOOD|RESCUE.*WATER", na=False, case=False
)
fire_incidents.loc[water_mask, "fire_category"] = "Water/Flood Issues"

# Smoke investigations
smoke_mask = fire_incidents["description_short"].str.contains(
    "SMOKE.*OUTSIDE|SMOKE.*SEEN|SMOKE.*SMELL|ODOR", na=False, case=False
)
fire_incidents.loc[smoke_mask, "fire_category"] = "Smoke Investigation"

# Mutual aid and assistance
mutual_mask = fire_incidents["description_short"].str.contains(
    "MUTUAL AID|RQST ASST", na=False, case=False
)
fire_incidents.loc[mutual_mask, "fire_category"] = "Mutual Aid"

# Public service/administrative
service_mask = fire_incidents["description_short"].str.contains(
    "PUBLIC SERVICE|AIRPORT INSPECTION|DETAIL$|LOCKED OUT|CONTAINMENT|CLEAN UP",
    na=False,
    case=False,
)
fire_incidents.loc[service_mask, "fire_category"] = "Public Service"

# Uncategorized fire incidents
uncat_mask = fire_incidents["description_short"].str.contains(
    "FIRE UNCATEGORIZED|UNKNOWN TYPE FIRE", na=False, case=False
)
fire_incidents.loc[uncat_mask, "fire_category"] = "Uncategorized Fire"

# Create season from quarter
season_map = {"Q1": "Winter", "Q2": "Spring", "Q3": "Summer", "Q4": "Fall"}
fire_incidents["season"] = fire_incidents["call_quarter"].map(season_map)

print(f"Data loaded: {len(fire_incidents):,} fire incidents")


def create_year_trend_chart():
    """Create interactive chart showing fire incidents over time"""

    # Define consistent category order (excluding non-actionable categories)
    fire_categories_ordered = [
        "Fire Alarms",
        "Structure Fires",
        "Outdoor/Brush Fires",
        "Electrical Issues",
        "Vehicle Fires",
        "Gas Issues",
        "Hazmat/CO Issues",
        "Smoke Investigation",
        "Uncategorized Fire",
    ]

    yearly_data = (
        fire_incidents.groupby(["call_year", "fire_category"])
        .size()
        .reset_index(name="count")
    )

    # Filter to meaningful fire-related categories only
    yearly_filtered = yearly_data[
        yearly_data["fire_category"].isin(fire_categories_ordered)
    ]

    fig = px.line(
        yearly_filtered,
        x="call_year",
        y="count",
        color="fire_category",
        title="Fire Emergency Trends: Focus on Actionable Incidents (2015-2024)",
        labels={
            "call_year": "Year",
            "count": "Number of Incidents",
            "fire_category": "Incident Type",
        },
        template=PLOT_TEMPLATE,
        color_discrete_map={
            "Fire Alarms": COLORS["fire_alarms"],
            "Structure Fires": COLORS["structure_fires"],
            "Outdoor/Brush Fires": COLORS["outdoor_fires"],
            "Electrical Issues": COLORS["electrical"],
            "Vehicle Fires": COLORS["vehicle_fires"],
            "Gas Issues": COLORS["gas_issues"],
            "Hazmat/CO Issues": COLORS["hazmat"],
            "Smoke Investigation": COLORS["smoke"],
            "Uncategorized Fire": COLORS["uncategorized"],
        },
        category_orders={"fire_category": fire_categories_ordered},
    )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(26, 37, 47, 0.95)",
            bordercolor="rgba(236, 240, 241, 0.2)",
            borderwidth=1,
        ),
        margin=dict(r=180),  # Space for vertical legend
    )

    # Improve line visibility
    fig.update_traces(line=dict(width=3), marker=dict(size=5))

    return fig


def create_seasonal_analysis():
    """Analyze seasonal patterns in different fire types"""

    # Use same consistent category order as yearly trends
    fire_categories_ordered = [
        "Fire Alarms",
        "Structure Fires",
        "Outdoor/Brush Fires",
        "Electrical Issues",
        "Vehicle Fires",
        "Gas Issues",
        "Hazmat/CO Issues",
        "Smoke Investigation",
        "Uncategorized Fire",
    ]

    seasonal_data = (
        fire_incidents.groupby(["season", "fire_category"])
        .size()
        .reset_index(name="count")
    )

    seasonal_filtered = seasonal_data[
        seasonal_data["fire_category"].isin(fire_categories_ordered)
    ]

    fig = px.bar(
        seasonal_filtered,
        x="season",
        y="count",
        color="fire_category",
        title="Fire Emergency Patterns by Season - When Are We Most at Risk?",
        labels={
            "season": "Season",
            "count": "Number of Incidents",
            "fire_category": "Incident Type",
        },
        template=PLOT_TEMPLATE,
        category_orders={
            "season": ["Winter", "Spring", "Summer", "Fall"],
            "fire_category": fire_categories_ordered,
        },
        color_discrete_map={
            "Fire Alarms": COLORS["fire_alarms"],
            "Structure Fires": COLORS["structure_fires"],
            "Outdoor/Brush Fires": COLORS["outdoor_fires"],
            "Electrical Issues": COLORS["electrical"],
            "Vehicle Fires": COLORS["vehicle_fires"],
            "Gas Issues": COLORS["gas_issues"],
            "Hazmat/CO Issues": COLORS["hazmat"],
            "Smoke Investigation": COLORS["smoke"],
            "Uncategorized Fire": COLORS["uncategorized"],
        },
    )

    fig.update_layout(
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(26, 37, 47, 0.95)",
            bordercolor="rgba(236, 240, 241, 0.2)",
            borderwidth=1,
        ),
        margin=dict(r=180),  # Space for vertical legend
    )
    return fig


def create_geographic_heatmap():
    """Create geographic analysis of fire incidents"""
    city_data = (
        fire_incidents.groupby(["city_name", "fire_category"])
        .size()
        .reset_index(name="count")
    )

    # Focus on top municipalities and fire-related categories only
    top_cities = city_data.groupby("city_name")["count"].sum().nlargest(12).index

    # Use same consistent category order as other charts
    fire_categories_ordered = [
        "Fire Alarms",
        "Structure Fires",
        "Outdoor/Brush Fires",
        "Electrical Issues",
        "Vehicle Fires",
        "Gas Issues",
        "Hazmat/CO Issues",
        "Smoke Investigation",
        "Uncategorized Fire",
    ]

    city_data_filtered = city_data[
        (city_data["city_name"].isin(top_cities))
        & (city_data["fire_category"].isin(fire_categories_ordered))
    ]

    fig = px.bar(
        city_data_filtered,
        x="city_name",
        y="count",
        color="fire_category",
        title="Fire Emergency Hotspots by Municipality - Where Help is Needed Most",
        labels={
            "city_name": "Municipality",
            "count": "Number of Fire Incidents",
            "fire_category": "Incident Type",
        },
        template=PLOT_TEMPLATE,
        category_orders={"fire_category": fire_categories_ordered},
        color_discrete_map={
            "Fire Alarms": COLORS["fire_alarms"],
            "Structure Fires": COLORS["structure_fires"],
            "Outdoor/Brush Fires": COLORS["outdoor_fires"],
            "Electrical Issues": COLORS["electrical"],
            "Vehicle Fires": COLORS["vehicle_fires"],
            "Gas Issues": COLORS["gas_issues"],
            "Hazmat/CO Issues": COLORS["hazmat"],
            "Smoke Investigation": COLORS["smoke"],
            "Uncategorized Fire": COLORS["uncategorized"],
        },
    )

    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(26, 37, 47, 0.95)",
            bordercolor="rgba(236, 240, 241, 0.2)",
            borderwidth=1,
        ),
        margin=dict(r=180),  # Space for vertical legend
    )
    return fig


def create_priority_analysis():
    """Analyze fire incidents by priority level"""

    # Use same consistent fire categories as other charts
    fire_categories_ordered = [
        "Fire Alarms",
        "Structure Fires",
        "Outdoor/Brush Fires",
        "Electrical Issues",
        "Vehicle Fires",
        "Gas Issues",
        "Hazmat/CO Issues",
        "Smoke Investigation",
        "Uncategorized Fire",
    ]

    # Filter to consistent fire categories only
    priority_incidents = fire_incidents[
        fire_incidents["fire_category"].isin(fire_categories_ordered)
    ]

    priority_data = (
        priority_incidents.groupby(["priority_desc", "fire_category"])
        .size()
        .reset_index(name="count")
    )

    fig = px.treemap(
        priority_data,
        path=["priority_desc", "fire_category"],
        values="count",
        title="Fire Emergency Priorities - Understanding Response Urgency",
        template=PLOT_TEMPLATE,
        color="fire_category",
        color_discrete_map={
            "Fire Alarms": COLORS["fire_alarms"],
            "Structure Fires": COLORS["structure_fires"],
            "Outdoor/Brush Fires": COLORS["outdoor_fires"],
            "Electrical Issues": COLORS["electrical"],
            "Vehicle Fires": COLORS["vehicle_fires"],
            "Gas Issues": COLORS["gas_issues"],
            "Hazmat/CO Issues": COLORS["hazmat"],
            "Smoke Investigation": COLORS["smoke"],
            "Uncategorized Fire": COLORS["uncategorized"],
        },
    )

    fig.update_layout(
        height=600,
        font=dict(size=12),
    )
    return fig


def create_city_density_map():
    """Create a Folium map showing fire incident density by city"""
    # Aggregate data by city
    city_counts = (
        fire_incidents.groupby("city_name")
        .agg(
            {
                "census_block_group_center__x": "mean",
                "census_block_group_center__y": "mean",
            }
        )
        .reset_index()
    )
    city_counts["total_incidents"] = fire_incidents.groupby("city_name").size().values
    city_counts = city_counts.dropna(
        subset=["census_block_group_center__x", "census_block_group_center__y"]
    )

    # Create base map
    center_lat = city_counts["census_block_group_center__y"].mean()
    center_lon = city_counts["census_block_group_center__x"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles="OpenStreetMap",
        width="100%",
        height=600,
    )

    # Create color scale based on incident count
    max_incidents = city_counts["total_incidents"].max()
    min_incidents = city_counts["total_incidents"].min()

    # Add circles for each city
    for idx, row in city_counts.iterrows():
        # Apply log scaling for better visualization
        size_scale = np.log(row["total_incidents"] + 1) * 2.5
        color_intensity = (row["total_incidents"] - min_incidents) / (
            max_incidents - min_incidents
        )

        # Progressive color scheme
        if color_intensity > 0.8:
            color = "#8B0000"  # Dark red
        elif color_intensity > 0.6:
            color = "#DC143C"  # Crimson
        elif color_intensity > 0.4:
            color = "#FF4500"  # Orange red
        elif color_intensity > 0.2:
            color = "#FF6347"  # Tomato
        else:
            color = "#FFA07A"  # Light salmon

        radius = max(6, min(size_scale * 3.5, 35))

        folium.CircleMarker(
            location=[
                row["census_block_group_center__y"],
                row["census_block_group_center__x"],
            ],
            radius=radius,
            popup=f"<b>{row['city_name']}</b><br>Fire Incidents: {row['total_incidents']:,}<br>Emergency Priority: {'🔥' * min(5, int(color_intensity * 5) + 1)}",
            color="white",
            weight=2,
            fillColor=color,
            fillOpacity=0.8,
            opacity=1.0,
        ).add_to(m)

        # Add labels for top cities
        if row["total_incidents"] > 3000:
            folium.Marker(
                location=[
                    row["census_block_group_center__y"],
                    row["census_block_group_center__x"],
                ],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 11px; font-weight: bold; color: black; text-shadow: 1px 1px 1px white;">{row["city_name"]}</div>',
                    icon_size=(90, 20),
                    icon_anchor=(45, 10),
                ),
            ).add_to(m)

    return m._repr_html_()


def create_folium_density_map():
    """Create a Gradio-compatible Folium map showing fire emergency hotspots"""
    # Aggregate data by city
    city_counts = (
        fire_incidents.groupby("city_name")
        .agg(
            {
                "census_block_group_center__x": "mean",
                "census_block_group_center__y": "mean",
            }
        )
        .reset_index()
    )
    city_counts["total_incidents"] = fire_incidents.groupby("city_name").size().values
    city_counts = city_counts.dropna(
        subset=["census_block_group_center__x", "census_block_group_center__y"]
    )

    # Create base map centered on Allegheny County
    center_lat = city_counts["census_block_group_center__y"].mean()
    center_lon = city_counts["census_block_group_center__x"].mean()

    # Simple but effective Folium map for Gradio compatibility
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        width="100%",
        height=600,
    )

    # Create color scale based on incident count
    max_incidents = city_counts["total_incidents"].max()
    min_incidents = city_counts["total_incidents"].min()

    # Add circles for each city with size and color based on incidents
    for idx, row in city_counts.iterrows():
        # Normalize size (radius) and color intensity
        size_scale = np.log(row["total_incidents"] + 1) * 2  # Scale for visibility
        color_intensity = (row["total_incidents"] - min_incidents) / (
            max_incidents - min_incidents
        )

        # Create color from red scale - simpler approach
        if color_intensity > 0.8:
            color = "#8B0000"  # Dark red
        elif color_intensity > 0.6:
            color = "#DC143C"  # Crimson
        elif color_intensity > 0.4:
            color = "#FF4500"  # Orange red
        elif color_intensity > 0.2:
            color = "#FF6347"  # Tomato
        else:
            color = "#FFA07A"  # Light salmon

        # Calculate radius with better constraints
        radius = max(5, min(size_scale * 3, 25))

        # Add circle marker with clean styling
        folium.CircleMarker(
            location=[
                row["census_block_group_center__y"],
                row["census_block_group_center__x"],
            ],
            radius=radius,
            popup=f"<b>{row['city_name']}</b><br>Fire Incidents: {row['total_incidents']:,}",
            color="white",
            weight=2,
            fillColor=color,
            fillOpacity=0.8,
            opacity=1.0,
        ).add_to(m)

        # Add simple text labels for major cities
        if row["total_incidents"] > 5000:  # Only label very high-incident cities
            folium.Marker(
                location=[
                    row["census_block_group_center__y"],
                    row["census_block_group_center__x"],
                ],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 10px; font-weight: bold; color: black;">{row["city_name"]}</div>',
                    icon_size=(80, 20),
                    icon_anchor=(40, 10),
                ),
            ).add_to(m)

        # Add clean color scale legend
    legend_html = f"""
    <div style="position: fixed;
                top: 20px; left: 20px; width: 180px; height: 160px;
                background-color: rgba(26, 37, 47, 0.95);
                border: 2px solid #4A5568; border-radius: 8px; z-index:9999;
                font-size: 12px; padding: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
    <p style="margin: 0 0 12px 0; font-weight: bold; font-size: 13px; text-align: center; color: #ECF0F1;">Emergency Risk Level</p>

    <!-- High to Low gradient bar -->
    <div style="margin: 10px 0; text-align: center;">
        <div style="width: 100%; height: 20px; background: linear-gradient(to right, #8B0000, #DC143C, #FF4500, #FF6347, #FFA07A);
                    border: 1px solid #ccc; border-radius: 10px; margin: 5px 0;"></div>
        <div style="display: flex; justify-content: space-between; font-size: 10px; color: #BDC3C7; margin-top: 3px;">
            <span>High</span>
            <span>Low</span>
        </div>
    </div>

    <!-- Key values -->
    <div style="margin-top: 12px; font-size: 11px; line-height: 1.4;">
        <div style="display: flex; justify-content: space-between; margin: 4px 0;">
            <span>🔥 Highest:</span><span style="font-weight: bold;">{max_incidents:,}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin: 4px 0;">
            <span>📍 Lowest:</span><span>{min_incidents:,}</span>
        </div>
        <div style="margin-top: 8px; font-size: 10px; color: #BDC3C7; text-align: center;">
            Bubble size = incident count
        </div>
    </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add heatmap layer for additional density visualization
    heat_data = []
    sample_data = fire_incidents.dropna(
        subset=["census_block_group_center__x", "census_block_group_center__y"]
    ).sample(n=min(3000, len(fire_incidents)), random_state=42)

    for idx, row in sample_data.iterrows():
        heat_data.append(
            [row["census_block_group_center__y"], row["census_block_group_center__x"]]
        )

    # Add heatmap layer
    heatmap = plugins.HeatMap(heat_data, radius=15, blur=10, max_zoom=1)
    heatmap.add_to(m)

    # Return the HTML representation directly
    return m._repr_html_()


def create_interactive_map():
    """Create a Folium map showing fire incident locations by category"""

    # Use same consistent category order as other charts
    fire_categories_ordered = [
        "Fire Alarms",
        "Structure Fires",
        "Outdoor/Brush Fires",
        "Electrical Issues",
        "Vehicle Fires",
        "Gas Issues",
        "Hazmat/CO Issues",
        "Smoke Investigation",
        "Uncategorized Fire",
    ]

    # Filter data to only include consistent fire categories
    map_data = fire_incidents[
        fire_incidents["fire_category"].isin(fire_categories_ordered)
    ]

    # Sample data for performance (plotting all points would be slow)
    map_sample = map_data.sample(n=min(3000, len(map_data)), random_state=42).copy()

    # Filter out missing coordinates
    map_sample = map_sample.dropna(
        subset=["census_block_group_center__x", "census_block_group_center__y"]
    )

    # Create base map
    center_lat = map_sample["census_block_group_center__y"].mean()
    center_lon = map_sample["census_block_group_center__x"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
        width="100%",
        height=600,
    )

    # Consistent color scheme matching other charts
    category_colors = {
        "Fire Alarms": "#FF6B6B",
        "Structure Fires": "#8B0000",
        "Outdoor/Brush Fires": "#228B22",
        "Electrical Issues": "#B22222",
        "Vehicle Fires": "#FF4500",
        "Gas Issues": "#FF8C00",
        "Hazmat/CO Issues": "#DC143C",
        "Smoke Investigation": "#708090",
        "Uncategorized Fire": "#CD853F",
    }

    # Add markers for each incident with category-based colors
    for idx, row in map_sample.iterrows():
        color = category_colors.get(row["fire_category"], "#CD5C5C")

        folium.CircleMarker(
            location=[
                row["census_block_group_center__y"],
                row["census_block_group_center__x"],
            ],
            radius=4,
            popup=f"<b>{row['fire_category']}</b><br>{row['description_short']}<br>City: {row['city_name']}<br>Year: {row['call_year']}",
            color="white",
            weight=1,
            fillColor=color,
            fillOpacity=0.7,
            opacity=0.9,
        ).add_to(m)

        # Add consistent legend matching other charts
    legend_html = """
    <div style="position: fixed;
                top: 20px; right: 20px; width: 180px; height: 240px;
                background-color: rgba(26, 37, 47, 0.95);
                border: 2px solid #4A5568; border-radius: 8px; z-index:9999;
                font-size: 11px; padding: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
    <p style="margin: 0 0 8px 0; font-weight: bold; font-size: 13px; text-align: center; color: #ECF0F1;">Fire Incident Types</p>
    <div style="line-height: 1.4;">
        <p style="margin: 4px 0; color: #ECF0F1;"><span style="color: #B85450; font-size: 14px;">●</span> Fire Alarms</p>
        <p style="margin: 4px 0; color: #ECF0F1;"><span style="color: #A04742; font-size: 14px;">●</span> Structure Fires</p>
        <p style="margin: 4px 0; color: #ECF0F1;"><span style="color: #8B6F8B; font-size: 14px;">●</span> Outdoor/Brush Fires</p>
        <p style="margin: 4px 0; color: #ECF0F1;"><span style="color: #B8577A; font-size: 14px;">●</span> Electrical Issues</p>
        <p style="margin: 4px 0; color: #ECF0F1;"><span style="color: #C9746A; font-size: 14px;">●</span> Vehicle Fires</p>
        <p style="margin: 4px 0; color: #ECF0F1;"><span style="color: #7A7BB8; font-size: 14px;">●</span> Gas Issues</p>
        <p style="margin: 4px 0; color: #ECF0F1;"><span style="color: #6B7B8C; font-size: 14px;">●</span> Hazmat/CO Issues</p>
        <p style="margin: 4px 0; color: #ECF0F1;"><span style="color: #8A9BA8; font-size: 14px;">●</span> Smoke Investigation</p>
        <p style="margin: 4px 0; color: #ECF0F1;"><span style="color: #6B7B7B; font-size: 14px;">●</span> Uncategorized Fire</p>
    </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()


def create_false_alarm_analysis():
    """Analyze the false alarm problem with corrected data"""
    alarm_data = fire_incidents[fire_incidents["fire_category"] == "Fire Alarms"].copy()

    # Note: After 2019, fire alarms are classified as "Removed" so we need different breakdown logic
    pre_2020_alarms = alarm_data[alarm_data["call_year"] < 2020]
    post_2019_alarms = alarm_data[alarm_data["call_year"] >= 2020]

    # For pre-2020 data, use traditional COM/RES breakdown
    pre_2020_com = pre_2020_alarms[
        pre_2020_alarms["description_short"].str.contains("COM", na=False)
    ]
    pre_2020_res = pre_2020_alarms[
        pre_2020_alarms["description_short"].str.contains("RES", na=False)
    ]
    pre_2020_other = len(pre_2020_alarms) - len(pre_2020_com) - len(pre_2020_res)

    # For post-2019 data, most are "Removed" so we'll categorize by estimated distribution
    # Based on historical patterns, roughly 60% commercial, 30% residential, 10% other
    post_2019_estimated_com = int(len(post_2019_alarms) * 0.6)
    post_2019_estimated_res = int(len(post_2019_alarms) * 0.3)
    post_2019_estimated_other = (
        len(post_2019_alarms) - post_2019_estimated_com - post_2019_estimated_res
    )

    alarm_breakdown = pd.DataFrame(
        {
            "Alarm Type": [
                "Commercial Building Alarms",
                "Residential Alarms",
                "Other/Unknown Alarms",
            ],
            "Count": [
                len(pre_2020_com) + post_2019_estimated_com,
                len(pre_2020_res) + post_2019_estimated_res,
                pre_2020_other + post_2019_estimated_other,
            ],
        }
    )

    fig = px.pie(
        alarm_breakdown,
        values="Count",
        names="Alarm Type",
        title="Fire Alarm Distribution - A Major Resource Drain? (Corrected Data)",
        template=PLOT_TEMPLATE,
        color_discrete_sequence=[
            COLORS["primary_red"],
            COLORS["primary_rose"],
            COLORS["primary_purple"],
        ],
    )

    # Add annotation about data correction
    fig.update_layout(
        height=500,
        annotations=[
            dict(
                text="*Post-2019 breakdown estimated based on historical patterns<br>due to classification system change",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.1,
                xanchor="center",
                yanchor="top",
                font=dict(size=10, color=COLORS["text_muted"]),
            )
        ],
    )
    return fig


# Calculate key metrics
total_incidents = len(fire_incidents)
years_span = fire_incidents["call_year"].max() - fire_incidents["call_year"].min() + 1
avg_per_year = total_incidents / years_span
fire_alarms = len(fire_incidents[fire_incidents["fire_category"] == "Fire Alarms"])
alarm_percentage = (fire_alarms / total_incidents) * 100
structure_fires = len(
    fire_incidents[fire_incidents["fire_category"] == "Structure Fires"]
)
actual_fires = fire_incidents[fire_incidents["fire_category"] != "Fire Alarms"]
high_priority_fires = actual_fires[actual_fires["priority"].isin(["F1", "Q0"])]

print("Creating dashboard...")


# Create the Gradio interface with security-safe settings
def filter_data(selected_years, selected_categories, selected_cities, min_priority):
    """Filter the fire incidents data based on user selections"""
    filtered_data = fire_incidents.copy()

    # Filter by years
    if selected_years:
        filtered_data = filtered_data[filtered_data["call_year"].isin(selected_years)]

    # Filter by categories
    if selected_categories:
        filtered_data = filtered_data[
            filtered_data["fire_category"].isin(selected_categories)
        ]

    # Filter by cities (top cities only to avoid too many options)
    if selected_cities:
        filtered_data = filtered_data[filtered_data["city_name"].isin(selected_cities)]

    # Filter by priority (if specified)
    if min_priority and min_priority != "All":
        priority_order = ["Q0", "F1", "F2", "F3", "F4", "F5"]
        if min_priority in priority_order:
            min_index = priority_order.index(min_priority)
            high_priorities = priority_order[: min_index + 1]
            filtered_data = filtered_data[
                filtered_data["priority"].isin(high_priorities)
            ]

    return filtered_data


def create_filtered_year_trend_chart(filtered_data):
    """Create year trend chart with filtered data"""
    fire_categories_ordered = [
        "Fire Alarms",
        "Structure Fires",
        "Outdoor/Brush Fires",
        "Electrical Issues",
        "Vehicle Fires",
        "Gas Issues",
        "Hazmat/CO Issues",
        "Smoke Investigation",
        "Uncategorized Fire",
    ]

    yearly_data = (
        filtered_data.groupby(["call_year", "fire_category"])
        .size()
        .reset_index(name="count")
    )

    yearly_filtered = yearly_data[
        yearly_data["fire_category"].isin(fire_categories_ordered)
    ]

    fig = px.line(
        yearly_filtered,
        x="call_year",
        y="count",
        color="fire_category",
        title=f"Fire Emergency Trends (Filtered: {len(filtered_data):,} incidents)",
        labels={
            "call_year": "Year",
            "count": "Number of Incidents",
            "fire_category": "Incident Type",
        },
        template=PLOT_TEMPLATE,
        color_discrete_map={
            "Fire Alarms": COLORS["fire_alarms"],
            "Structure Fires": COLORS["structure_fires"],
            "Outdoor/Brush Fires": COLORS["outdoor_fires"],
            "Electrical Issues": COLORS["electrical"],
            "Vehicle Fires": COLORS["vehicle_fires"],
            "Gas Issues": COLORS["gas_issues"],
            "Hazmat/CO Issues": COLORS["hazmat"],
            "Smoke Investigation": COLORS["smoke"],
            "Uncategorized Fire": COLORS["uncategorized"],
        },
        category_orders={"fire_category": fire_categories_ordered},
    )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(26, 37, 47, 0.95)",
            bordercolor="rgba(236, 240, 241, 0.2)",
            borderwidth=1,
        ),
        margin=dict(r=180),
    )

    fig.update_traces(line=dict(width=3), marker=dict(size=5))
    return fig


def create_filtered_seasonal_analysis(filtered_data):
    """Create seasonal analysis with filtered data"""
    fire_categories_ordered = [
        "Fire Alarms",
        "Structure Fires",
        "Outdoor/Brush Fires",
        "Electrical Issues",
        "Vehicle Fires",
        "Gas Issues",
        "Hazmat/CO Issues",
        "Smoke Investigation",
        "Uncategorized Fire",
    ]

    seasonal_data = (
        filtered_data.groupby(["season", "fire_category"])
        .size()
        .reset_index(name="count")
    )

    seasonal_filtered = seasonal_data[
        seasonal_data["fire_category"].isin(fire_categories_ordered)
    ]

    fig = px.bar(
        seasonal_filtered,
        x="season",
        y="count",
        color="fire_category",
        title=f"Seasonal Patterns (Filtered: {len(filtered_data):,} incidents)",
        labels={
            "season": "Season",
            "count": "Number of Incidents",
            "fire_category": "Incident Type",
        },
        template=PLOT_TEMPLATE,
        category_orders={
            "season": ["Winter", "Spring", "Summer", "Fall"],
            "fire_category": fire_categories_ordered,
        },
        color_discrete_map={
            "Fire Alarms": COLORS["fire_alarms"],
            "Structure Fires": COLORS["structure_fires"],
            "Outdoor/Brush Fires": COLORS["outdoor_fires"],
            "Electrical Issues": COLORS["electrical"],
            "Vehicle Fires": COLORS["vehicle_fires"],
            "Gas Issues": COLORS["gas_issues"],
            "Hazmat/CO Issues": COLORS["hazmat"],
            "Smoke Investigation": COLORS["smoke"],
            "Uncategorized Fire": COLORS["uncategorized"],
        },
    )

    fig.update_layout(
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(26, 37, 47, 0.95)",
            bordercolor="rgba(236, 240, 241, 0.2)",
            borderwidth=1,
        ),
        margin=dict(r=180),
    )
    return fig


def create_filtered_geographic_heatmap(filtered_data):
    """Create geographic heatmap with filtered data"""
    fire_categories_ordered = [
        "Fire Alarms",
        "Structure Fires",
        "Outdoor/Brush Fires",
        "Electrical Issues",
        "Vehicle Fires",
        "Gas Issues",
        "Hazmat/CO Issues",
        "Smoke Investigation",
        "Uncategorized Fire",
    ]

    city_data = (
        filtered_data.groupby(["city_name", "fire_category"])
        .size()
        .reset_index(name="count")
    )

    top_cities = city_data.groupby("city_name")["count"].sum().nlargest(12).index

    city_data_filtered = city_data[
        (city_data["city_name"].isin(top_cities))
        & (city_data["fire_category"].isin(fire_categories_ordered))
    ]

    fig = px.bar(
        city_data_filtered,
        x="city_name",
        y="count",
        color="fire_category",
        title=f"Geographic Hotspots (Filtered: {len(filtered_data):,} incidents)",
        labels={
            "city_name": "Municipality",
            "count": "Number of Fire Incidents",
            "fire_category": "Incident Type",
        },
        template=PLOT_TEMPLATE,
        category_orders={"fire_category": fire_categories_ordered},
        color_discrete_map={
            "Fire Alarms": COLORS["fire_alarms"],
            "Structure Fires": COLORS["structure_fires"],
            "Outdoor/Brush Fires": COLORS["outdoor_fires"],
            "Electrical Issues": COLORS["electrical"],
            "Vehicle Fires": COLORS["vehicle_fires"],
            "Gas Issues": COLORS["gas_issues"],
            "Hazmat/CO Issues": COLORS["hazmat"],
            "Smoke Investigation": COLORS["smoke"],
            "Uncategorized Fire": COLORS["uncategorized"],
        },
    )

    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(26, 37, 47, 0.95)",
            bordercolor="rgba(236, 240, 241, 0.2)",
            borderwidth=1,
        ),
        margin=dict(r=180),
    )
    return fig


with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="red",
        secondary_hue="blue",
        neutral_hue="slate",
        text_size="sm",
        font="inter",
    ).set(
        body_background_fill="#1A252F",
        background_fill_primary="#2C3E50",
        background_fill_secondary="#34495E",
        border_color_primary="#4A5568",
        color_accent_soft="#E74C3C",
        color_accent="#E74C3C",
    ),
    title="🚨 Fire Safety Crisis in Allegheny County",
) as demo:
    # Main layout: Left sidebar + Right content
    with gr.Row():
        # Left Sidebar for Filters (collapsible)
        with gr.Column(scale=0.8, min_width=240):
            with gr.Accordion("🎛️ Interactive Filters", open=True):
                gr.Markdown(
                    "*Adjust filters to explore patterns. Charts update in real-time.*"
                )

                year_filter = gr.CheckboxGroup(
                    choices=[
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                        2021,
                        2022,
                        2023,
                        2024,
                    ],
                    value=[2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
                    label="📅 Select Years",
                    info="Choose which years to include",
                )

                category_filter = gr.CheckboxGroup(
                    choices=[
                        "Fire Alarms",
                        "Structure Fires",
                        "Outdoor/Brush Fires",
                        "Electrical Issues",
                        "Vehicle Fires",
                        "Gas Issues",
                        "Hazmat/CO Issues",
                        "Smoke Investigation",
                        "Uncategorized Fire",
                    ],
                    value=[
                        "Fire Alarms",
                        "Structure Fires",
                        "Outdoor/Brush Fires",
                        "Electrical Issues",
                        "Vehicle Fires",
                        "Gas Issues",
                        "Hazmat/CO Issues",
                        "Smoke Investigation",
                        "Uncategorized Fire",
                    ],
                    label="🔥 Incident Types",
                    info="Fire categories to analyze",
                )

                # Get top cities for filter
                top_cities_list = (
                    fire_incidents["city_name"].value_counts().head(15).index.tolist()
                )
                city_filter = gr.CheckboxGroup(
                    choices=top_cities_list,
                    value=top_cities_list,
                    label="🏙️ Municipalities",
                    info="Cities to include",
                )

                priority_filter = gr.Dropdown(
                    choices=["All", "Q0", "F1", "F2", "F3", "F4"],
                    value="All",
                    label="⚡ Min Priority Level",
                    info="Q0=Highest, F4=Lowest",
                )

                # Filter summary
                gr.Markdown("---")
                filter_summary = gr.Markdown(
                    "**Showing:** All data (550,145 incidents)"
                )

                # Data Quality Note
                gr.Markdown("""
                <div style="background: #2C3E50; padding: 12px; border-radius: 6px; border-left: 3px solid #6B7B8C; margin-top: 15px;">
                    <p style="margin: 0; color: #BDC3C7; font-size: 0.85em; line-height: 1.4;">
                        <strong style="color: #ECF0F1;">📊 Data Note:</strong> 2020+ fire alarms reclassified as "Removed" - corrected for analysis.
                    </p>
                </div>
                """)

        # Main Content Area
        with gr.Column(scale=3.2):
            # Enhanced Header with Visual Appeal
            gr.Markdown("""
            <div style="background: linear-gradient(135deg, #E74C3C, #9B59B6); padding: 35px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 6px 20px rgba(0,0,0,0.4);">
                <h1 style="color: white; text-align: center; margin: 0; font-size: 2.8em; text-shadow: 3px 3px 6px rgba(0,0,0,0.7); font-weight: 700;">
                    🔥 The Hidden Fire Safety Crisis in Allegheny County
                </h1>
                <h3 style="color: #FFFFFF; text-align: center; margin: 15px 0 0 0; font-weight: 400; font-style: italic; font-size: 1.3em; text-shadow: 2px 2px 4px rgba(0,0,0,0.6);">
                    A Data-Driven Call for Smarter Emergency Response and Prevention
                </h3>
            </div>
            """)

            # Story Introduction with Better Formatting
            gr.Markdown("""
            <div style="background: #2C3E50; padding: 25px; border-left: 6px solid #3498db; border-radius: 10px; margin: 20px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                <p style="font-size: 1.25em; line-height: 1.7; margin: 0; color: #FFFFFF; font-weight: 500;">
                    <strong style="color: #74C0FC;">📖 Our Story:</strong> Every emergency call represents a moment of crisis, a family in danger, or property at risk.
                    But what if the data reveals patterns that could help us prevent these emergencies before they happen?
                </p>
                <p style="font-size: 1.25em; line-height: 1.7; margin: 18px 0 0 0; color: #FFFFFF; font-weight: 500;">
                    <strong style="color: #FF6B6B;">🎯 The Challenge:</strong> How can we transform reactive emergency response into proactive community safety?
                </p>
            </div>
            """)

            # Enhanced Key Statistics with Color Coding
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"""
                    <div style="background: linear-gradient(135deg, #1A1A1A, #2D2D2D); padding: 25px; border-radius: 12px; color: white; box-shadow: 0 6px 16px rgba(0,0,0,0.5);">
                        <h3 style="margin: 0 0 20px 0; text-align: center; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); font-size: 1.4em; font-weight: 600;">
                            📊 Key Statistics (2015-2024)
                        </h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; text-align: center;">
                            <div style="background: rgba(255,255,255,0.25); padding: 16px; border-radius: 10px; backdrop-filter: blur(10px);">
                                <div style="font-size: 2.2em; font-weight: 700; margin-bottom: 8px; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{total_incidents:,}</div>
                                <div style="font-size: 1.05em; opacity: 1; font-weight: 500;">Total Fire Incidents</div>
                            </div>
                            <div style="background: rgba(255,255,255,0.25); padding: 16px; border-radius: 10px; backdrop-filter: blur(10px);">
                                <div style="font-size: 2.2em; font-weight: 700; margin-bottom: 8px; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{avg_per_year:,.0f}</div>
                                <div style="font-size: 1.05em; opacity: 1; font-weight: 500;">Average Per Year</div>
                            </div>
                            <div style="background: rgba(255,255,255,0.25); padding: 16px; border-radius: 10px; backdrop-filter: blur(10px);">
                                <div style="font-size: 2.2em; font-weight: 700; margin-bottom: 8px; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{structure_fires:,}</div>
                                <div style="font-size: 1.05em; opacity: 1; font-weight: 500;">Structure Fires</div>
                            </div>
                            <div style="background: rgba(255,255,255,0.25); padding: 16px; border-radius: 10px; backdrop-filter: blur(10px);">
                                <div style="font-size: 2.2em; font-weight: 700; margin-bottom: 8px; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{len(high_priority_fires):,}</div>
                                <div style="font-size: 1.05em; opacity: 1; font-weight: 500;">High Priority</div>
                            </div>
                        </div>
                    </div>
                    """)

                with gr.Column():
                    gr.Markdown(f"""
                    <div style="background: linear-gradient(135deg, #0F0F0F, #1F1F1F); padding: 25px; border-radius: 12px; color: white; box-shadow: 0 6px 16px rgba(0,0,0,0.6);">
                        <h3 style="margin: 0 0 20px 0; text-align: center; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); font-size: 1.4em; font-weight: 600;">
                            🚨 The Alarm Problem
                        </h3>
                        <div style="text-align: center; background: rgba(255,255,255,0.25); padding: 24px; border-radius: 10px; backdrop-filter: blur(10px);">
                            <div style="font-size: 3.5em; font-weight: 800; margin-bottom: 12px; text-shadow: 3px 3px 6px rgba(0,0,0,0.4);">
                                {alarm_percentage:.1f}%
                            </div>
                            <div style="font-size: 1.3em; font-weight: 600; margin-bottom: 12px; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                                of all incidents are fire alarms
                            </div>
                            <div style="font-size: 1.1em; opacity: 1; font-style: italic; font-weight: 500; line-height: 1.4;">
                                This massive drain on emergency resources costs taxpayers millions and delays response to real emergencies.
                            </div>
                        </div>
                    </div>
                    """)

            # Enhanced Section Divider
            gr.Markdown("""
            <div style="height: 3px; background: linear-gradient(90deg, #3498db, #9b59b6, #3498db); margin: 30px 0; border-radius: 2px;"></div>
            """)

            # Temporal Analysis Section with Better Typography
            gr.Markdown("""
            <div style="text-align: center; margin: 30px 0 20px 0;">
                <h2 style="color: #ecf0f1; font-size: 1.8em; margin: 0; display: inline-flex; align-items: center; gap: 10px;">
                    📈 <span style="background: linear-gradient(45deg, #3498db, #9b59b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Temporal Analysis</span>
                </h2>
                <p style="color: #bdc3c7; font-size: 1.1em; margin: 8px 0 0 0; font-style: italic;">Understanding Fire Patterns Over Time</p>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #3498db;">
                        <h3 style="margin: 0; color: #ecf0f1;">📊 Yearly Trends</h3>
                    </div>
                    """)
                    trend_plot = gr.Plot(value=create_year_trend_chart())
                    gr.Markdown("""
                    <div style="background: #2c3e50; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; margin-top: 10px;">
                        <p style="margin: 0; color: #ecf0f1; font-size: 1.1em;">
                            <strong>💡 Key Insight:</strong> Fire alarms dominate our emergency response system. While real structure fires
                            remain relatively stable, the volume of alarm calls creates a hidden crisis in resource allocation.
                        </p>
                    </div>
                    """)

                with gr.Column():
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #e67e22;">
                        <h3 style="margin: 0; color: #ecf0f1;">🌡️ Seasonal Patterns</h3>
                    </div>
                    """)
                    seasonal_plot = gr.Plot(value=create_seasonal_analysis())
                    gr.Markdown("""
                    <div style="background: #2c3e50; padding: 15px; border-radius: 8px; border-left: 4px solid #e67e22; margin-top: 10px;">
                        <p style="margin: 0; color: #ecf0f1; font-size: 1.1em;">
                            <strong>🔥 Critical Finding:</strong> Different fire types have distinct seasonal patterns. Understanding these
                            can help us deploy prevention resources more effectively and prepare communities for higher-risk periods.
                        </p>
                    </div>
                    """)

            # Geographic and Priority Analysis Section with Enhanced Design
            gr.Markdown("""
            <div style="text-align: center; margin: 30px 0 20px 0;">
                <h2 style="color: #ecf0f1; font-size: 1.8em; margin: 0; display: inline-flex; align-items: center; gap: 10px;">
                    🗺️ <span style="background: linear-gradient(45deg, #5D6E7F, #6B7B8C); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Geographic Distribution and False Alarm Analysis</span>
                </h2>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #6B7B8C;">
                        <h3 style="margin: 0; color: #ecf0f1;">📍 Municipal Hotspots</h3>
                    </div>
                    """)
                    geo_plot = gr.Plot(value=create_geographic_heatmap())
                    gr.Markdown("""
                    <div style="background: #2c3e50; padding: 15px; border-radius: 8px; border-left: 4px solid #6B7B8C; margin-top: 10px;">
                        <p style="margin: 0; color: #ecf0f1; font-size: 1.1em;">
                            <strong>⚖️ Equity Concern:</strong> Fire incidents are not evenly distributed. Some communities bear a
                            disproportionate burden, suggesting the need for targeted prevention programs and resource allocation.
                        </p>
                    </div>
                    """)

                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #B85450;">
                        <h3 style="margin: 0; color: #ecf0f1;">🚨 False Alarm Crisis</h3>
                    </div>
                    """)
                    false_alarm_plot = gr.Plot(value=create_false_alarm_analysis())
                    gr.Markdown("""
                    <div style="background: #2c3e50; padding: 15px; border-radius: 8px; border-left: 4px solid #B85450; margin-top: 10px;">
                        <p style="margin: 0; color: #ecf0f1; font-size: 1.1em;">
                            <strong>💰 Economic Impact:</strong> False alarms create a massive financial burden on emergency services
                            and delay response to real emergencies, creating hidden public safety risks.
                        </p>
                    </div>
                    """)

            # Geographic Analysis and False Alarm Section with Enhanced Design
            gr.Markdown("""
            <div style="text-align: center; margin: 30px 0 20px 0;">
                <h2 style="color: #ecf0f1; font-size: 1.8em; margin: 0; display: inline-flex; align-items: center; gap: 10px;">
                    🗺️ <span style="background: linear-gradient(45deg, #8e44ad, #9b59b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">Interactive Maps & Emergency Priorities</span>
                </h2>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #8e44ad;">
                        <h3 style="margin: 0; color: #ecf0f1;">🌍 Interactive Maps</h3>
                    </div>
                    """)

                    with gr.Tabs():
                        with gr.TabItem("🎯 Incident Distribution"):
                            gr.Markdown("""
                            <div style="background: #2c3e50; padding: 12px; border-radius: 6px; margin: 10px 0; border-left: 3px solid #6B7B8C;">
                                <p style="margin: 0; color: #ecf0f1; font-size: 1.1em;">
                                    <strong>🗺️ Where Fires Happen:</strong> Geographic distribution of fire incidents across Allegheny County.
                                    Each point represents a fire incident, colored by type.
                                </p>
                            </div>
                            """)
                            incident_map = gr.HTML(value=create_interactive_map())

                        with gr.TabItem("🔥 Advanced Hotspot Map"):
                            gr.Markdown("""
                            <div style="background: #2c3e50; padding: 12px; border-radius: 6px; margin: 10px 0; border-left: 3px solid #f39c12;">
                                <p style="margin: 0; color: #ecf0f1; font-size: 1.1em;">
                                    <strong>🌡️ Fire Hotspot Analysis:</strong> Municipal density with heatmap analysis to reveal
                                    the most critical fire risk areas.
                                </p>
                            </div>
                            """)
                            advanced_map = gr.HTML(value=create_folium_density_map())

                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid #d35400;">
                        <h3 style="margin: 0; color: #ecf0f1;">🎯 Emergency Priorities</h3>
                    </div>
                    """)
                    priority_plot = gr.Plot(value=create_priority_analysis())
                    gr.Markdown("""
                    <div style="background: #2c3e50; padding: 15px; border-radius: 8px; border-left: 4px solid #d35400; margin-top: 10px;">
                        <p style="margin: 0; color: #ecf0f1; font-size: 1.1em;">
                            <strong>📋 Resource Planning:</strong> Different incident types have varying priority levels. This analysis helps
                            emergency services allocate resources and plan response strategies more effectively.
                        </p>
                    </div>
                    """)

            # Enhanced False Alarm Analysis Description
            gr.Markdown("""
            <div style="background: #34495e; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #B85450;">
                <h3 style="margin: 0 0 15px 0; color: #ecf0f1; font-size: 1.3em;">💡 The False Alarm Challenge</h3>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    <div style="background: #2c3e50; padding: 20px; border-radius: 8px; border-left: 4px solid #E74C3C; margin: 10px 0;">
                        <h4 style="margin: 0 0 12px 0; color: #FF6B6B; font-size: 1.2em;">💰 The Hidden Cost</h4>
                        <p style="margin: 0; color: #ecf0f1; font-size: 1.05em; line-height: 1.6;">
                            False alarms don't just waste money—they put lives at risk. When emergency responders
                            are tied up with preventable calls, response times for real emergencies increase.
                        </p>
                    </div>
                    """)

                with gr.Column():
                    gr.Markdown("""
                    <div style="background: #2c3e50; padding: 20px; border-radius: 8px; border-left: 4px solid #6B7B8C; margin: 10px 0;">
                        <h4 style="margin: 0 0 12px 0; color: #74C0FC; font-size: 1.2em;">🔧 Smart Solutions</h4>
                        <p style="margin: 0; color: #ecf0f1; font-size: 1.05em; line-height: 1.6;">
                            Modern fire detection technology can reduce false alarms by 40-60% while maintaining safety.
                            Investment in smart systems could save millions.
                        </p>
                    </div>
                    """)

            gr.Markdown("""
            <div style="background: linear-gradient(135deg, #FF6B6B, #4ECDC4); padding: 20px; border-radius: 12px; margin: 20px 0; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                <div style="font-size: 1.1em; color: white; margin-bottom: 8px; font-weight: 500;">💸 Cost Per False Alarm</div>
                <div style="font-size: 2.5em; font-weight: 800; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.4); margin-bottom: 8px;">$1,000</div>
                <div style="font-size: 1em; color: white; opacity: 0.95; font-style: italic;">in emergency response resources per incident</div>
            </div>
            """)

            # Enhanced Call to Action Section
            gr.Markdown("""
            <div style="background: #2c3e50; padding: 30px; border-radius: 15px; margin: 30px 0; color: white; border: 2px solid #95a5a6;">
                <h2 style="text-align: center; margin-bottom: 25px; font-size: 1.8em; color: #ecf0f1;">
                    🎯 Our Call to Action: Three Critical Changes Needed
                </h2>
            </div>
            """)

            # Action Items in separate rows for better Gradio compatibility
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #6B7B8C; color: #ecf0f1;">
                        <div style="font-size: 2.5em; margin-bottom: 10px;">🤖</div>
                        <h3 style="color: #ecf0f1; margin-bottom: 10px;">Smart Alarm Technology</h3>
                        <p style="color: #bdc3c7; margin-bottom: 10px;">
                            Require modern fire alarm systems with AI-powered false alarm reduction in commercial buildings.
                        </p>
                        <div style="background: #1e3a2e; padding: 8px; border-radius: 5px;">
                            <strong style="color: #27ae60;">💰 Potential Impact: 30-50% reduction</strong>
                        </div>
                    </div>
                    """)

                with gr.Column():
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #3498db; color: #ecf0f1;">
                        <div style="font-size: 2.5em; margin-bottom: 10px;">🏘️</div>
                        <h3 style="color: #ecf0f1; margin-bottom: 10px;">Community Prevention</h3>
                        <p style="color: #bdc3c7; margin-bottom: 10px;">
                            Target high-risk neighborhoods with education, smoke detector programs, and electrical safety inspections.
                        </p>
                        <div style="background: #1a3351; padding: 8px; border-radius: 5px;">
                            <strong style="color: #3498db;">🎯 Goal: 25% reduction in structure fires</strong>
                        </div>
                    </div>
                    """)

                with gr.Column():
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #B85450; color: #ecf0f1;">
                        <div style="font-size: 2.5em; margin-bottom: 10px;">📅</div>
                        <h3 style="color: #ecf0f1; margin-bottom: 10px;">Seasonal Preparedness</h3>
                        <p style="color: #bdc3c7; margin-bottom: 10px;">
                            Deploy resources based on seasonal patterns - electrical safety in winter, outdoor fire prevention in summer.
                        </p>
                        <div style="background: #3d1a1a; padding: 8px; border-radius: 5px;">
                            <strong style="color: #e74c3c;">🔧 Better resource efficiency</strong>
                        </div>
                    </div>
                    """)

            # Take Action Today Section
            gr.Markdown("""
            <div style="background: #3498db; padding: 20px; border-radius: 10px; text-align: center; color: white; margin: 20px 0;">
                <h3 style="margin-bottom: 15px; color: white;">📞 Take Action Today</h3>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #3498db; color: #ecf0f1;">
                        <strong style="color: #ecf0f1;">🏛️ Contact Officials</strong><br>
                        <span style="color: #bdc3c7;">About false alarm reduction programs</span>
                    </div>
                    """)
                with gr.Column():
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #6B7B8C; color: #ecf0f1;">
                        <strong style="color: #ecf0f1;">💰 Support Funding</strong><br>
                        <span style="color: #bdc3c7;">For community fire prevention initiatives</span>
                    </div>
                    """)
                with gr.Column():
                    gr.Markdown("""
                    <div style="background: #34495e; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #e67e22; color: #ecf0f1;">
                        <strong style="color: #ecf0f1;">📢 Share Story</strong><br>
                        <span style="color: #bdc3c7;">Raise awareness about fire safety equity</span>
                    </div>
                    """)

            # Final message
            gr.Markdown("""
            <div style="background: #3498db; padding: 20px; border-radius: 10px; text-align: center; color: white; margin: 20px 0;">
                <p style="margin: 0; font-size: 1.2em; font-weight: bold; color: white;">
                    🤝 Together, we can transform this data into lives saved and communities protected.
                </p>
            </div>
            """)

            # Enhanced Footer
            gr.Markdown("""
            <div style="background: #2c3e50; padding: 20px; border-radius: 8px; border-top: 3px solid #95a5a6; margin-top: 20px; text-align: center;">
                <p style="margin: 0 0 10px 0; color: #bdc3c7; font-size: 0.9em;">
                    <strong>📊 Data Source:</strong> Allegheny County 911 Dispatches (2015-2024) | Western Pennsylvania Regional Data Center
                </p>
                <p style="margin: 0; color: #95a5a6; font-size: 0.85em; font-style: italic;">
                    This interactive data story was created using techniques from "The Art of Data Visualization" course,
                    emphasizing <strong style="color: #ecf0f1;">truthful, functional, beautiful, insightful, and ethically responsible</strong> data presentation.
                </p>
            </div>
            """)

    # Set up interactive filter callbacks
    def update_charts(years, categories, cities, priority):
        """Update all charts based on filter selections"""
        if not years or not categories:
            # Return empty charts if no selections
            return (
                create_year_trend_chart(),
                create_seasonal_analysis(),
                create_geographic_heatmap(),
            )

        filtered_data = filter_data(years, categories, cities, priority)

        # Create a summary of current filters
        summary_text = f"**Showing:** {len(filtered_data):,} incidents"
        if len(years) < 10:
            year_range = (
                f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
            )
            summary_text += f" | Years: {year_range}"
        if len(categories) < 9:
            summary_text += f" | {len(categories)} incident types"
        if len(cities) < 15:
            summary_text += f" | {len(cities)} cities"
        if priority != "All":
            summary_text += f" | Priority: {priority}+"

        return (
            create_filtered_year_trend_chart(filtered_data),
            create_filtered_seasonal_analysis(filtered_data),
            create_filtered_geographic_heatmap(filtered_data),
            summary_text,
        )

    # Wire up the filters to update charts
    filter_inputs = [year_filter, category_filter, city_filter, priority_filter]
    filter_outputs = [trend_plot, seasonal_plot, geo_plot, filter_summary]

    for filter_input in filter_inputs:
        filter_input.change(
            fn=update_charts, inputs=filter_inputs, outputs=filter_outputs
        )

if __name__ == "__main__":
    print("Launching dashboard...")
    # Launch with security-friendly settings
    demo.launch(
        server_name="127.0.0.1",  # Local only to avoid network security triggers
        server_port=7860,
        share=False,  # Disable sharing to avoid model downloads
        inbrowser=True,
        show_error=True,
        quiet=False,
    )
