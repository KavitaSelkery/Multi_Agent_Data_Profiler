"""
Visualization tools for data quality analysis
"""
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json
from datetime import datetime
from loguru import logger

class VisualizationTools:
    """Tools for creating data quality visualizations"""
    
    def __init__(self):
        """Initialize visualization tools"""
        self.color_palette = {
            'critical': '#FF5252',
            'high': '#FF7043',
            'medium': '#FFB74D',
            'low': '#81C784',
            'info': '#29B6F6',
            'success': '#66BB6A',
            'warning': '#FFA726',
            'error': '#EF5350'
        }
    
    def create_null_heatmap(self, df: pd.DataFrame, title: str = "Null Value Heatmap") -> go.Figure:
        """Create heatmap of null values"""
        try:
            # Create null matrix
            null_matrix = df.isnull().astype(int)
            
            fig = px.imshow(
                null_matrix.T,
                title=title,
                labels=dict(x="Row Index", y="Column", color="Is Null"),
                color_continuous_scale="Reds",
                aspect="auto"
            )
            
            fig.update_layout(
                height=max(400, len(df.columns) * 20),
                width=800,
                title_x=0.5,
                xaxis_title="Row Index",
                yaxis_title="Columns",
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating null heatmap: {str(e)}")
            return self._create_empty_figure("Error creating null heatmap")
    
    def create_data_type_distribution(self, schema_df: pd.DataFrame, title: str = "Data Type Distribution") -> go.Figure:
        """Create pie chart of data type distribution"""
        try:
            if schema_df.empty or 'DATA_TYPE' not in schema_df.columns:
                return self._create_empty_figure("No schema data available")
            
            type_counts = schema_df['DATA_TYPE'].value_counts()
            
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title=title,
                hole=0.3,
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            
            fig.update_layout(
                height=500,
                title_x=0.5,
                showlegend=True,
                font=dict(size=12)
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="Type: %{label}<br>Count: %{value}<br>Percentage: %{percent}"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating data type distribution: {str(e)}")
            return self._create_empty_figure("Error creating data type distribution")
    
    def create_numeric_distributions(self, df: pd.DataFrame, max_cols: int = 6) -> go.Figure:
        """Create subplots of numeric column distributions"""
        try:
            # Get numeric columns
            numeric_cols = [
                col for col in df.columns 
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 1
            ]
            
            if not numeric_cols:
                return self._create_empty_figure("No numeric columns found")
            
            # Limit number of columns
            numeric_cols = numeric_cols[:max_cols]
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, 
                cols=n_cols,
                subplot_titles=numeric_cols,
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            for idx, col in enumerate(numeric_cols):
                row = (idx // n_cols) + 1
                col_pos = (idx % n_cols) + 1
                
                # Clean data
                clean_data = df[col].dropna()
                
                # Add histogram
                fig.add_trace(
                    go.Histogram(
                        x=clean_data,
                        name=col,
                        nbinsx=min(50, len(clean_data)),
                        marker_color=self.color_palette['info'],
                        opacity=0.7
                    ),
                    row=row, col=col_pos
                )
                
                # Add box plot on secondary axis
                fig.add_trace(
                    go.Box(
                        x=clean_data,
                        name=col,
                        marker_color=self.color_palette['warning'],
                        boxpoints='outliers',
                        jitter=0.3,
                        pointpos=-1.8
                    ),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                height=n_rows * 300,
                title_text="Numeric Column Distributions",
                title_x=0.5,
                showlegend=False,
                font=dict(size=12)
            )
            
            # Update axes
            for i in range(1, len(numeric_cols) + 1):
                fig.update_xaxes(title_text="Value", row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
                fig.update_yaxes(title_text="Frequency", row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating numeric distributions: {str(e)}")
            return self._create_empty_figure("Error creating numeric distributions")
    
    def create_categorical_analysis(self, df: pd.DataFrame, max_cols: int = 4) -> go.Figure:
        """Create categorical column analysis"""
        try:
            # Get categorical columns
            categorical_cols = [
                col for col in df.columns 
                if df[col].dtype == 'object' or df[col].nunique() < 20
            ]
            
            if not categorical_cols:
                return self._create_empty_figure("No categorical columns found")
            
            # Limit number of columns
            categorical_cols = categorical_cols[:max_cols]
            n_cols = min(2, len(categorical_cols))
            n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, 
                cols=n_cols,
                subplot_titles=[f"{col} (Unique: {df[col].nunique()})" for col in categorical_cols],
                specs=[[{"type": "bar"} for _ in range(n_cols)] for _ in range(n_rows)],
                vertical_spacing=0.15
            )
            
            for idx, col in enumerate(categorical_cols):
                row = (idx // n_cols) + 1
                col_pos = (idx % n_cols) + 1
                
                value_counts = df[col].value_counts().head(10)  # Top 10 values
                
                fig.add_trace(
                    go.Bar(
                        x=value_counts.index.astype(str),
                        y=value_counts.values,
                        name=col,
                        marker_color=px.colors.qualitative.Set3[idx % len(px.colors.qualitative.Set3)],
                        text=value_counts.values,
                        textposition='auto'
                    ),
                    row=row, col=col_pos
                )
                
                # Rotate x-axis labels for readability
                fig.update_xaxes(tickangle=45, row=row, col=col_pos)
            
            fig.update_layout(
                height=n_rows * 400,
                title_text="Categorical Column Analysis (Top 10 Values)",
                title_x=0.5,
                showlegend=False,
                font=dict(size=12),
                bargap=0.2
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating categorical analysis: {str(e)}")
            return self._create_empty_figure("Error creating categorical analysis")
    
    def create_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation matrix heatmap"""
        try:
            # Get numeric columns
            numeric_cols = [
                col for col in df.columns 
                if pd.api.types.is_numeric_dtype(df[col])
            ]
            
            if len(numeric_cols) < 2:
                return self._create_empty_figure("Need at least 2 numeric columns for correlation")
            
            numeric_df = df[numeric_cols]
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            corr_matrix_masked = corr_matrix.mask(mask)
            
            fig = px.imshow(
                corr_matrix_masked,
                title="Correlation Matrix",
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.index,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto"
            )
            
            # Add annotations
            for i, row in enumerate(corr_matrix.index):
                for j, col in enumerate(corr_matrix.columns):
                    if i <= j:  # Only show upper triangle
                        continue
                    
                    value = corr_matrix.iloc[i, j]
                    if not pd.isna(value):
                        fig.add_annotation(
                            x=j,
                            y=i,
                            text=f"{value:.2f}",
                            showarrow=False,
                            font=dict(
                                color="white" if abs(value) > 0.5 else "black",
                                size=10
                            )
                        )
            
            fig.update_layout(
                height=600,
                width=800,
                title_x=0.5,
                xaxis_title="Columns",
                yaxis_title="Columns",
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {str(e)}")
            return self._create_empty_figure("Error creating correlation matrix")
    
    def create_temporal_analysis(self, df: pd.DataFrame, date_column: str) -> go.Figure:
        """Create temporal analysis of date column"""
        try:
            if date_column not in df.columns:
                return self._create_empty_figure(f"Column '{date_column}' not found")
            
            # Convert to datetime
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            temporal_df = df.dropna(subset=[date_column]).copy()
            
            if temporal_df.empty:
                return self._create_empty_figure(f"No valid dates in '{date_column}'")
            
            # Extract date components
            temporal_df['year'] = temporal_df[date_column].dt.year
            temporal_df['month'] = temporal_df[date_column].dt.month
            temporal_df['day'] = temporal_df[date_column].dt.day
            temporal_df['day_of_week'] = temporal_df[date_column].dt.dayofweek
            temporal_df['hour'] = temporal_df[date_column].dt.hour
            
            # Create subplots
            fig = make_subplots(
                rows=2, 
                cols=2,
                subplot_titles=[
                    'Records by Year',
                    'Records by Month',
                    'Records by Day of Week',
                    'Records by Hour'
                ],
                vertical_spacing=0.15
            )
            
            # Year distribution
            year_counts = temporal_df['year'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=year_counts.index,
                    y=year_counts.values,
                    name='Year',
                    marker_color=self.color_palette['info']
                ),
                row=1, col=1
            )
            
            # Month distribution
            month_counts = temporal_df['month'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=month_counts.index,
                    y=month_counts.values,
                    name='Month',
                    marker_color=self.color_palette['success']
                ),
                row=1, col=2
            )
            
            # Day of week distribution
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = temporal_df['day_of_week'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=[day_names[i] for i in day_counts.index],
                    y=day_counts.values,
                    name='Day of Week',
                    marker_color=self.color_palette['warning']
                ),
                row=2, col=1
            )
            
            # Hour distribution
            hour_counts = temporal_df['hour'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=hour_counts.index,
                    y=hour_counts.values,
                    name='Hour',
                    marker_color=self.color_palette['error']
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                title_text=f"Temporal Analysis - {date_column}",
                title_x=0.5,
                showlegend=False,
                font=dict(size=12)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating temporal analysis: {str(e)}")
            return self._create_empty_figure("Error creating temporal analysis")
    
    def create_data_quality_dashboard(self, quality_metrics: Dict[str, Any]) -> go.Figure:
        """Create data quality dashboard visualization"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, 
                cols=2,
                subplot_titles=[
                    'Data Quality Score',
                    'Issue Severity Distribution',
                    'Null Percentage by Column',
                    'Data Type Distribution'
                ],
                specs=[
                    [{"type": "indicator"}, {"type": "pie"}],
                    [{"type": "bar"}, {"type": "pie"}]
                ],
                vertical_spacing=0.2,
                horizontal_spacing=0.2
            )
            
            # 1. Quality Score Gauge
            quality_score = quality_metrics.get('overall_score', 0)
            gauge_color = self._get_quality_color(quality_score)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=quality_score,
                    title={'text': "Quality Score"},
                    domain={'row': 0, 'column': 0},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 60], 'color': self.color_palette['error']},
                            {'range': [60, 80], 'color': self.color_palette['warning']},
                            {'range': [80, 100], 'color': self.color_palette['success']}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': quality_score
                        }
                    }
                ),
                row=1, col=1
            )
            
            # 2. Issue Severity Distribution
            issues_by_severity = quality_metrics.get('issues_by_severity', {})
            if issues_by_severity:
                severities = list(issues_by_severity.keys())
                counts = list(issues_by_severity.values())
                colors = [self.color_palette.get(sev, '#CCCCCC') for sev in severities]
                
                fig.add_trace(
                    go.Pie(
                        labels=severities,
                        values=counts,
                        name="Issues by Severity",
                        marker=dict(colors=colors),
                        hole=0.4,
                        textinfo='label+percent'
                    ),
                    row=1, col=2
                )
            
            # 3. Null Percentage by Column (top 10)
            null_percentages = quality_metrics.get('null_percentages', {})
            if null_percentages:
                columns = list(null_percentages.keys())[:10]
                percentages = list(null_percentages.values())[:10]
                
                # Create colors based on percentage
                colors = []
                for pct in percentages:
                    if pct > 30:
                        colors.append(self.color_palette['critical'])
                    elif pct > 10:
                        colors.append(self.color_palette['warning'])
                    else:
                        colors.append(self.color_palette['success'])
                
                fig.add_trace(
                    go.Bar(
                        x=columns,
                        y=percentages,
                        name='Null %',
                        marker_color=colors,
                        text=[f"{p:.1f}%" for p in percentages],
                        textposition='auto'
                    ),
                    row=2, col=1
                )
                
                fig.update_xaxes(tickangle=45, row=2, col=1)
                fig.update_yaxes(title_text="Null Percentage (%)", row=2, col=1)
            
            # 4. Data Type Distribution
            data_types = quality_metrics.get('data_type_distribution', {})
            if data_types:
                fig.add_trace(
                    go.Pie(
                        labels=list(data_types.keys()),
                        values=list(data_types.values()),
                        name="Data Types",
                        hole=0.3,
                        textinfo='label+percent'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                title_text="Data Quality Dashboard",
                title_x=0.5,
                font=dict(size=12),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating quality dashboard: {str(e)}")
            return self._create_empty_figure("Error creating quality dashboard")
    
    def create_anomaly_detection_plot(self, df: pd.DataFrame, anomalies: List[Dict[str, Any]]) -> go.Figure:
        """Create anomaly detection visualization"""
        try:
            # Get first numeric column with anomalies
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_cols:
                return self._create_empty_figure("No numeric columns for anomaly detection")
            
            target_col = numeric_cols[0]
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add normal points
            normal_mask = ~df.index.isin([a.get('index') for a in anomalies if 'index' in a])
            fig.add_trace(
                go.Scatter(
                    x=df.index[normal_mask],
                    y=df[target_col][normal_mask],
                    mode='markers',
                    name='Normal',
                    marker=dict(
                        color=self.color_palette['success'],
                        size=8,
                        opacity=0.6
                    )
                )
            )
            
            # Add anomaly points
            anomaly_indices = [a.get('index') for a in anomalies if 'index' in a and a.get('index') in df.index]
            if anomaly_indices:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_indices,
                        y=df.loc[anomaly_indices, target_col],
                        mode='markers',
                        name='Anomaly',
                        marker=dict(
                            color=self.color_palette['critical'],
                            size=12,
                            symbol='x',
                            line=dict(width=2, color='black')
                        ),
                        text=[a.get('reason', 'Anomaly') for a in anomalies if 'index' in a],
                        hoverinfo='text+x+y'
                    )
                )
            
            # Add mean line
            mean_val = df[target_col].mean()
            fig.add_hline(
                y=mean_val,
                line_dash="dash",
                line_color=self.color_palette['info'],
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="bottom right"
            )
            
            # Add ±2σ lines
            std_val = df[target_col].std()
            fig.add_hline(
                y=mean_val + 2 * std_val,
                line_dash="dot",
                line_color=self.color_palette['warning'],
                annotation_text=f"+2σ: {mean_val + 2 * std_val:.2f}"
            )
            fig.add_hline(
                y=mean_val - 2 * std_val,
                line_dash="dot",
                line_color=self.color_palette['warning'],
                annotation_text=f"-2σ: {mean_val - 2 * std_val:.2f}"
            )
            
            fig.update_layout(
                height=600,
                title_text=f"Anomaly Detection - {target_col}",
                title_x=0.5,
                xaxis_title="Index",
                yaxis_title=target_col,
                font=dict(size=12),
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating anomaly plot: {str(e)}")
            return self._create_empty_figure("Error creating anomaly plot")
    
    def create_comparison_plot(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                             column: str, title: str = "Data Comparison") -> go.Figure:
        """Create comparison plot between two datasets"""
        try:
            if column not in df1.columns or column not in df2.columns:
                return self._create_empty_figure(f"Column '{column}' not found in both datasets")
            
            fig = go.Figure()
            
            # Add histograms
            fig.add_trace(
                go.Histogram(
                    x=df1[column].dropna(),
                    name='Dataset 1',
                    opacity=0.7,
                    marker_color=self.color_palette['info'],
                    nbinsx=50
                )
            )
            
            fig.add_trace(
                go.Histogram(
                    x=df2[column].dropna(),
                    name='Dataset 2',
                    opacity=0.7,
                    marker_color=self.color_palette['success'],
                    nbinsx=50
                )
            )
            
            fig.update_layout(
                height=500,
                title_text=title,
                title_x=0.5,
                xaxis_title=column,
                yaxis_title="Frequency",
                font=dict(size=12),
                barmode='overlay',
                hovermode='x unified'
            )
            
            # Add summary statistics
            stats_text = f"""
            <b>Dataset 1 ({len(df1)} rows):</b><br>
            Mean: {df1[column].mean():.2f}<br>
            Std: {df1[column].std():.2f}<br>
            Min: {df1[column].min():.2f}<br>
            Max: {df1[column].max():.2f}<br><br>
            
            <b>Dataset 2 ({len(df2)} rows):</b><br>
            Mean: {df2[column].mean():.2f}<br>
            Std: {df2[column].std():.2f}<br>
            Min: {df2[column].min():.2f}<br>
            Max: {df2[column].max():.2f}
            """
            
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comparison plot: {str(e)}")
            return self._create_empty_figure("Error creating comparison plot")
    
    def export_to_html(self, fig: go.Figure, filename: str = "visualization.html") -> str:
        """Export figure to HTML file"""
        try:
            html_content = fig.to_html(
                full_html=True,
                include_plotlyjs='cdn',
                config={
                    'displayModeBar': True,
                    'responsive': True,
                    'displaylogo': False
                }
            )
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Exported visualization to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting to HTML: {str(e)}")
            return ""
    
    def get_figure_as_base64(self, fig: go.Figure) -> str:
        """Convert figure to base64 encoded string"""
        try:
            # Convert to image bytes
            img_bytes = fig.to_image(format="png", width=800, height=600)
            
            # Encode to base64
            base64_str = base64.b64encode(img_bytes).decode('utf-8')
            
            return f"data:image/png;base64,{base64_str}"
            
        except Exception as e:
            logger.error(f"Error converting figure to base64: {str(e)}")
            return ""
    
    def _get_quality_color(self, score: float) -> str:
        """Get color based on quality score"""
        if score >= 80:
            return self.color_palette['success']
        elif score >= 60:
            return self.color_palette['warning']
        else:
            return self.color_palette['error']
    
    def _create_empty_figure(self, message: str = "No data available") -> go.Figure:
        """Create empty figure with message"""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16, color="gray"),
            xref="paper",
            yref="paper"
        )
        
        fig.update_layout(
            height=400,
            width=600,
            plot_bgcolor='white',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def create_summary_statistics_table(self, df: pd.DataFrame) -> go.Figure:
        """Create table of summary statistics"""
        try:
            statistics = []
            
            for column in df.columns:
                col_data = df[column]
                
                stats = {
                    'Column': column,
                    'Type': str(col_data.dtype),
                    'Non-Null': col_data.count(),
                    'Null': col_data.isnull().sum(),
                    'Null %': f"{(col_data.isnull().sum() / len(col_data) * 100):.1f}%",
                    'Unique': col_data.nunique()
                }
                
                if pd.api.types.is_numeric_dtype(col_data):
                    stats.update({
                        'Mean': f"{col_data.mean():.2f}",
                        'Std': f"{col_data.std():.2f}",
                        'Min': f"{col_data.min():.2f}",
                        '25%': f"{col_data.quantile(0.25):.2f}",
                        '50%': f"{col_data.quantile(0.50):.2f}",
                        '75%': f"{col_data.quantile(0.75):.2f}",
                        'Max': f"{col_data.max():.2f}"
                    })
                
                statistics.append(stats)
            
            # Create table
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=list(statistics[0].keys()),
                    fill_color=self.color_palette['info'],
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[[stats.get(key, '') for stats in statistics] for key in statistics[0].keys()],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=11)
                )
            )])
            
            fig.update_layout(
                height=min(800, len(statistics) * 30 + 100),
                title_text="Summary Statistics",
                title_x=0.5,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating statistics table: {str(e)}")
            return self._create_empty_figure("Error creating statistics table")