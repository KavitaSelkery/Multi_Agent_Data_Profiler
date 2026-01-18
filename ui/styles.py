"""
UI Styles and Theme Configuration
"""
from typing import Dict, Any, Optional
from enum import Enum
import streamlit as st

class ColorPalette(str, Enum):
    """Color palette for UI"""
    PRIMARY = "#1E88E5"
    SECONDARY = "#FFC107"
    SUCCESS = "#4CAF50"
    WARNING = "#FF9800"
    ERROR = "#F44336"
    INFO = "#2196F3"
    DARK = "#212121"
    LIGHT = "#F5F5F5"
    
class Theme(str, Enum):
    """UI Themes"""
    LIGHT = "light"
    DARK = "dark"

class UIStyles:
    """Manages UI styles and themes"""
    
    @staticmethod
    def apply_custom_styles(theme: Theme = Theme.LIGHT):
        """Apply custom CSS styles"""
        primary_color = ColorPalette.PRIMARY
        secondary_color = ColorPalette.SECONDARY
        background_color = "#FFFFFF" if theme == Theme.LIGHT else "#121212"
        text_color = "#333333" if theme == Theme.LIGHT else "#FFFFFF"
        
        styles = f"""
        <style>
            /* General styles */
            .main {{
                background-color: {background_color};
                color: {text_color};
            }}
            
            /* Headers */
            h1, h2, h3, h4, h5, h6 {{
                color: {primary_color} !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            
            /* Cards */
            .card {{
                background-color: {'#F8F9FA' if theme == Theme.LIGHT else '#1E1E1E'};
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                border-left: 4px solid {primary_color};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .critical-card {{
                border-left-color: {ColorPalette.ERROR};
                background-color: {'#FFEBEE' if theme == Theme.LIGHT else '#2D1B1B'};
            }}
            
            .warning-card {{
                border-left-color: {ColorPalette.WARNING};
                background-color: {'#FFF3E0' if theme == Theme.LIGHT else '#2D271B'};
            }}
            
            .success-card {{
                border-left-color: {ColorPalette.SUCCESS};
                background-color: {'#E8F5E9' if theme == Theme.LIGHT else '#1B2D1C'};
            }}
            
            /* Buttons */
            .stButton > button {{
                background-color: {primary_color};
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                transition: all 0.3s;
            }}
            
            .stButton > button:hover {{
                background-color: {secondary_color};
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }}
            
            /* Metrics */
            .metric {{
                background: linear-gradient(135deg, {primary_color}, {secondary_color});
                color: white;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin: 5px;
            }}
            
            .metric-value {{
                font-size: 2.5rem;
                font-weight: bold;
                margin: 10px 0;
            }}
            
            .metric-label {{
                font-size: 0.9rem;
                opacity: 0.9;
            }}
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 24px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                background-color: transparent;
                border-radius: 4px 4px 0px 0px;
                padding: 10px 20px;
                font-weight: bold;
            }}
            
            /* DataFrames */
            .dataframe {{
                border-radius: 5px;
                overflow: hidden;
            }}
            
            .dataframe th {{
                background-color: {primary_color} !important;
                color: white !important;
                font-weight: bold !important;
            }}
            
            .dataframe tr:nth-child(even) {{
                background-color: {'#F5F5F5' if theme == Theme.LIGHT else '#2D2D2D'} !important;
            }}
            
            /* Status indicators */
            .status-critical {{
                color: {ColorPalette.ERROR};
                font-weight: bold;
                animation: pulse 2s infinite;
            }}
            
            .status-warning {{
                color: {ColorPalette.WARNING};
                font-weight: bold;
            }}
            
            .status-success {{
                color: {ColorPalette.SUCCESS};
                font-weight: bold;
            }}
            
            @keyframes pulse {{
                0% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
                100% {{ opacity: 1; }}
            }}
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: {'#F1F1F1' if theme == Theme.LIGHT else '#2D2D2D'};
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: {primary_color};
                border-radius: 4px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: {secondary_color};
            }}
            
            /* Tooltips */
            .tooltip {{
                position: relative;
                display: inline-block;
            }}
            
            .tooltip .tooltiptext {{
                visibility: hidden;
                width: 200px;
                background-color: {ColorPalette.DARK};
                color: white;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
            }}
            
            .tooltip:hover .tooltiptext {{
                visibility: visible;
                opacity: 1;
            }}
        </style>
        """
        
        st.markdown(styles, unsafe_allow_html=True)
    
    @staticmethod
    def get_icon_for_severity(severity: str) -> str:
        """Get icon for severity level"""
        icons = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢',
            'info': '🔵',
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'loading': '⏳'
        }
        return icons.get(severity.lower(), '📝')
    
    @staticmethod
    def get_color_for_score(score: float) -> str:
        """Get color based on score"""
        if score >= 80:
            return ColorPalette.SUCCESS
        elif score >= 60:
            return ColorPalette.WARNING
        else:
            return ColorPalette.ERROR
    
    @staticmethod
    def create_data_quality_badge(score: float, size: str = "medium") -> str:
        """Create a badge for data quality score"""
        color = UIStyles.get_color_for_score(score)
        badge_size = {
            "small": "height: 25px; width: 25px; font-size: 0.8rem;",
            "medium": "height: 40px; width: 40px; font-size: 1rem;",
            "large": "height: 60px; width: 60px; font-size: 1.5rem;"
        }.get(size, "height: 40px; width: 40px; font-size: 1rem;")
        
        return f"""
        <div style="
            background-color: {color};
            color: white;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            {badge_size}
        ">
            {int(score)}
        </div>
        """
    
    @staticmethod
    def create_progress_bar(value: float, max_value: float = 100, height: int = 20) -> str:
        """Create a custom progress bar"""
        percentage = (value / max_value) * 100
        color = UIStyles.get_color_for_score(percentage)
        
        return f"""
        <div style="
            width: 100%;
            background-color: #E0E0E0;
            border-radius: 10px;
            overflow: hidden;
            height: {height}px;
        ">
            <div style="
                width: {percentage}%;
                background-color: {color};
                height: 100%;
                border-radius: 10px;
                transition: width 0.5s ease-in-out;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: {height * 0.6}px;
            ">
                {value:.1f}
            </div>
        </div>
        """