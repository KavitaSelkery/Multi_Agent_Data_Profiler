"""
UI Layout Components
"""
from typing import Dict, Any, List, Optional, Callable
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from ui.styles import UIStyles, ColorPalette

class LayoutComponents:
    """Reusable UI layout components"""
    
    @staticmethod
    def create_header(title: str, subtitle: str = None, icon: str = None):
        """Create page header"""
        col1, col2 = st.columns([1, 4])
        with col1:
            if icon:
                st.markdown(f"<h1 style='text-align: center;'>{icon}</h1>", unsafe_allow_html=True)
        with col2:
            st.title(title)
            if subtitle:
                st.caption(subtitle)
        
        st.divider()
    
    @staticmethod
    def create_metric_card(title: str, value: Any, change: Optional[float] = None, 
                          icon: str = "📊", help_text: str = None):
        """Create a metric card"""
        change_color = ColorPalette.SUCCESS if change and change >= 0 else ColorPalette.ERROR
        change_icon = "↗️" if change and change >= 0 else "↘️" if change else ""
        
        with st.container():
            st.markdown(f"""
            <div class="card">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 1.5rem; margin-right: 10px;">{icon}</span>
                    <span style="font-weight: bold; font-size: 1.1rem;">{title}</span>
                    {f'<span style="margin-left: auto; color: {change_color}; font-weight: bold;">{change_icon} {abs(change):.1f}%</span>' if change is not None else ''}
                </div>
                <div style="font-size: 2.5rem; font-weight: bold; color: {ColorPalette.PRIMARY};">
                    {value}
                </div>
                {f'<div style="font-size: 0.9rem; color: #666; margin-top: 5px;">{help_text}</div>' if help_text else ''}
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def create_data_quality_metrics(metrics: Dict[str, float]):
        """Create data quality metrics dashboard"""
        cols = st.columns(len(metrics))
        
        for idx, (metric_name, metric_value) in enumerate(metrics.items()):
            with cols[idx]:
                LayoutComponents.create_metric_card(
                    title=metric_name.replace('_', ' ').title(),
                    value=f"{metric_value:.1f}%",
                    icon=UIStyles.get_icon_for_severity(
                        'success' if metric_value >= 80 else 'warning' if metric_value >= 60 else 'critical'
                    ),
                    help_text=f"Data {metric_name}"
                )
    
    @staticmethod
    def create_issue_card(issue: Dict[str, Any], show_actions: bool = True):
        """Create a data quality issue card"""
        severity = issue.get('severity', 'info').lower()
        icon = UIStyles.get_icon_for_severity(severity)
        timestamp = issue.get('detected_at', datetime.now().isoformat())
        
        with st.container():
            card_class = {
                'critical': 'critical-card',
                'high': 'critical-card',
                'medium': 'warning-card',
                'low': 'card',
                'info': 'card'
            }.get(severity, 'card')
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; align-items: start; justify-content: space-between;">
                    <div>
                        <div style="font-weight: bold; font-size: 1.2rem; margin-bottom: 5px;">
                            {icon} {issue.get('title', 'Unknown Issue')}
                        </div>
                        <div style="color: #666; margin-bottom: 10px;">
                            Detected: {timestamp}
                        </div>
                    </div>
                    <span class="status-{severity}">
                        {severity.upper()}
                    </span>
                </div>
                <div style="margin: 10px 0;">
                    {issue.get('description', 'No description available')}
                </div>
                <div style="font-size: 0.9rem; color: #777;">
                    Table: {issue.get('table_name', 'N/A')} | Column: {issue.get('column_name', 'N/A')} | 
                    Records: {issue.get('record_count', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if show_actions:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📋 View Details", key=f"view_{issue.get('issue_id')}"):
                        st.session_state['selected_issue'] = issue
                with col2:
                    if st.button("👤 Assign", key=f"assign_{issue.get('issue_id')}"):
                        st.session_state['assign_issue'] = issue
                with col3:
                    if st.button("✅ Resolve", key=f"resolve_{issue.get('issue_id')}"):
                        st.session_state['resolve_issue'] = issue
    
    @staticmethod
    def create_sql_editor(default_sql: str = "SELECT * FROM your_table LIMIT 100", 
                         height: int = 200):
        """Create SQL editor component"""
        st.markdown("### SQL Editor")
        sql_code = st.text_area(
            "Write your SQL query:",
            value=default_sql,
            height=height,
            help="Write SQL queries to analyze your data"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            run_query = st.button("▶️ Run Query", type="primary")
        with col2:
            explain_query = st.button("📖 Explain Query")
        with col3:
            optimize_query = st.button("⚡ Optimize Query")
        
        return sql_code, run_query, explain_query, optimize_query
    
    @staticmethod
    def create_results_table(df: pd.DataFrame, max_rows: int = 100):
        """Create a styled results table"""
        if df.empty:
            st.info("No results to display")
            return
        
        st.markdown(f"**Results ({len(df)} rows, {len(df.columns)} columns)**")
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        with col3:
            st.metric("Columns", len(df.columns))
        
        # Display the dataframe
        st.dataframe(df, use_container_width=True)
        
        # Column information
        with st.expander("📊 Column Information"):
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                col_info.append({
                    'Column': col,
                    'Type': dtype,
                    'Nulls': null_count,
                    'Unique Values': unique_count,
                    'Null %': f"{(null_count / len(df) * 100):.1f}%" if len(df) > 0 else "0%"
                })
            
            col_info_df = pd.DataFrame(col_info)
            st.dataframe(col_info_df, use_container_width=True)
    
    @staticmethod
    def create_visualization_selector(data: pd.DataFrame):
        """Create visualization type selector"""
        viz_types = {
            "📈 Line Chart": "line",
            "📊 Bar Chart": "bar",
            "🎯 Scatter Plot": "scatter",
            "📦 Box Plot": "box",
            "📉 Histogram": "histogram",
            "🫓 Pie Chart": "pie",
            "🔥 Heatmap": "heatmap"
        }
        
        selected_viz = st.selectbox(
            "Select Visualization Type:",
            options=list(viz_types.keys())
        )
        
        # Based on visualization type, show additional options
        viz_type = viz_types[selected_viz]
        
        if viz_type in ["line", "bar", "scatter"]:
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("X Axis:", data.columns)
            with col2:
                y_column = st.selectbox("Y Axis:", data.columns)
            
            if viz_type == "scatter":
                color_column = st.selectbox("Color by (optional):", ["None"] + list(data.columns))
                color_column = None if color_column == "None" else color_column
                return viz_type, x_column, y_column, color_column
            return viz_type, x_column, y_column
        
        elif viz_type == "histogram":
            column = st.selectbox("Column:", data.columns)
            bins = st.slider("Number of bins:", 5, 100, 20)
            return viz_type, column, bins
        
        elif viz_type == "pie":
            column = st.selectbox("Column:", data.columns)
            return viz_type, column
        
        elif viz_type == "heatmap":
            columns = st.multiselect("Select columns:", data.select_dtypes(include=['number']).columns)
            return viz_type, columns
        
        return viz_type, None
    
    @staticmethod
    def create_agent_selector(agents: List[Dict[str, Any]], current_agent: str = None):
        """Create agent selector component"""
        st.markdown("### 🤖 Agent Selection")
        
        agent_options = {agent['name']: agent['id'] for agent in agents}
        selected_agent_name = st.selectbox(
            "Select Agent:",
            options=list(agent_options.keys()),
            index=list(agent_options.keys()).index(current_agent) if current_agent in agent_options else 0
        )
        
        selected_agent_id = agent_options[selected_agent_name]
        
        # Display agent details
        selected_agent = next((a for a in agents if a['id'] == selected_agent_id), None)
        if selected_agent:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Role", selected_agent.get('role', 'N/A'))
                st.metric("Team", selected_agent.get('team', 'N/A'))
            with col2:
                st.metric("Workload", f"{selected_agent.get('current_workload', 0)}/{selected_agent.get('max_workload', 10)}")
                st.metric("Skills", len(selected_agent.get('skills', [])))
        
        return selected_agent_id
    
    @staticmethod
    def create_loading_spinner(message: str = "Processing..."):
        """Create a loading spinner"""
        return st.spinner(message)
    
    @staticmethod
    def create_tabbed_interface(tabs: List[str], default_tab: str = None):
        """Create a tabbed interface"""
        return st.tabs(tabs)
    
    @staticmethod
    def create_footer():
        """Create page footer"""
        st.divider()
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>Data Quality Agent System | Powered by LangChain & Streamlit</p>
            <p style="font-size: 0.8rem;">© 2024 All rights reserved</p>
        </div>
        """, unsafe_allow_html=True)