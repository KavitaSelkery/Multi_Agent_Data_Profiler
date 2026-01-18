"""
Reporting agent for generating comprehensive reports
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
from langchain_core.tools import Tool
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from io import BytesIO
import base64
import sys, os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.base_agent import BaseAgent
from tools.snowflake_tools import SnowflakeManager
from tools.data_tools import DataAnalyzer
from config.prompts import REPORTER_AGENT_PROMPT

class ReporterAgent(BaseAgent):
    """Agent for generating data quality reports"""
    
    def __init__(self, snowflake_manager: SnowflakeManager):
        """Initialize reporter agent"""
        self.snowflake_manager = snowflake_manager
        self.data_analyzer = DataAnalyzer()
        
        # Define agent tools
        tools = [
            self.generate_summary_report_tool(),
            self.generate_detailed_report_tool(),
            self.generate_visualizations_tool(),
            self.export_report_tool(),
            self.create_action_plan_tool()
        ]
        
        super().__init__(
            name="Reporting Agent",
            description="Generates comprehensive data quality reports with visualizations",
            tools=tools
        )
    
    def generate_summary_report_tool(self) -> Tool:
        """Tool to generate summary report"""
        
        def generate_summary_report(table_name: str) -> str:
            """Generate executive summary report for a table"""
            try:
                # Get basic table information
                row_count = self.snowflake_manager.get_row_count(table_name)
                schema_df = self.snowflake_manager.get_table_schema(table_name)
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=5000)
                
                if sample_df.empty:
                    return f"No data found in table: {table_name}"
                
                # Analyze data quality
                quality_issues = self.data_analyzer.analyze_data_quality(sample_df, schema_df)
                
                # Calculate quality score
                quality_score = self.data_analyzer._calculate_quality_score(quality_issues)
                
                # Generate summary
                summary = f"""
                # 📊 Data Quality Executive Summary
                
                **Table:** {table_name}
                **Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## 📈 Key Metrics
                - **Total Rows:** {row_count:,}
                - **Total Columns:** {len(schema_df) if not schema_df.empty else len(sample_df.columns)}
                - **Data Quality Score:** {quality_score:.1f}/100
                - **Total Issues:** {len(quality_issues)}
                
                ## 🚨 Critical Findings
                """
                
                # Add critical issues
                critical_issues = [issue for issue in quality_issues if issue.get('severity') == 'critical']
                if critical_issues:
                    for issue in critical_issues[:5]:  # Show top 5 critical issues
                        summary += f"\n- **{issue['issue_type'].replace('_', ' ').title()}** in {issue['column']}: {issue.get('description', '')}"
                else:
                    summary += "\n- No critical issues found"
                
                summary += "\n\n## 🎯 Recommendations"
                
                # Generate recommendations
                recommendations = self.data_analyzer._generate_recommendations(quality_issues)
                for rec in recommendations[:5]:  # Show top 5 recommendations
                    summary += f"\n- {rec}"
                
                return summary
                
            except Exception as e:
                logger.error(f"Summary report error: {str(e)}")
                return f"Error generating summary report: {str(e)}"
        
        return Tool(
            name="generate_summary_report",
            func=generate_summary_report,
            description="Generate executive summary report with key metrics and findings"
        )
    
    def generate_detailed_report_tool(self) -> Tool:
        """Tool to generate detailed report"""
        
        def generate_detailed_report(table_name: str) -> str:
            """Generate detailed data quality report"""
            try:
                # Get data
                row_count = self.snowflake_manager.get_row_count(table_name)
                schema_df = self.snowflake_manager.get_table_schema(table_name)
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=10000)
                
                if sample_df.empty:
                    return f"No data found in table: {table_name}"
                
                # Comprehensive analysis
                analysis = self.data_analyzer.comprehensive_analysis(sample_df, schema_df)
                
                # Generate report
                report = f"""
                # 📋 Detailed Data Quality Report
                
                **Table:** {table_name}
                **Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                **Rows Analyzed:** {min(10000, row_count):,} (sampled from {row_count:,} total)
                
                ## 1. Executive Summary
                - **Overall Quality Score:** {analysis.get('data_quality_score', 0):.1f}/100
                - **Total Issues:** {len(analysis.get('data_quality_issues', []))}
                - **Memory Usage:** {analysis.get('memory_usage_mb', 0):.2f} MB
                
                ## 2. Data Quality Issues
                """
                
                # Group issues by type
                issues_by_type = {}
                for issue in analysis.get('data_quality_issues', []):
                    issue_type = issue['issue_type']
                    if issue_type not in issues_by_type:
                        issues_by_type[issue_type] = []
                    issues_by_type[issue_type].append(issue)
                
                for issue_type, issues in issues_by_type.items():
                    report += f"\n### {issue_type.replace('_', ' ').title()} ({len(issues)} issues)"
                    for issue in issues[:3]:  # Show top 3 of each type
                        report += f"\n- **{issue['column']}**: {issue.get('description', '')}"
                    if len(issues) > 3:
                        report += f"\n- ... and {len(issues) - 3} more"
                
                report += "\n\n## 3. Column Statistics\n"
                
                # Add column statistics
                stats = analysis.get('statistical_summary', {})
                for column, col_stats in list(stats.items())[:10]:  # Show first 10 columns
                    report += f"\n### {column}"
                    report += f"\n- Type: {col_stats.get('dtype', 'N/A')}"
                    report += f"\n- Null Count: {col_stats.get('null_count', 0):,}"
                    report += f"\n- Unique Values: {col_stats.get('unique_count', 0):,}"
                    
                    if 'min' in col_stats:
                        report += f"\n- Range: {col_stats.get('min'):.2f} to {col_stats.get('max'):.2f}"
                        report += f"\n- Mean: {col_stats.get('mean', 0):.2f}"
                        report += f"\n- Std Dev: {col_stats.get('std', 0):.2f}"
                
                if len(stats) > 10:
                    report += f"\n\n... and {len(stats) - 10} more columns"
                
                report += "\n\n## 4. Recommendations & Action Plan\n"
                
                # Add recommendations
                for rec in analysis.get('recommendations', []):
                    report += f"\n- {rec}"
                
                # Add data governance suggestions
                report += "\n\n### Data Governance Suggestions:"
                report += "\n1. Implement data validation rules"
                report += "\n2. Set up automated monitoring"
                report += "\n3. Establish data quality SLA"
                report += "\n4. Create data stewardship program"
                
                return report
                
            except Exception as e:
                logger.error(f"Detailed report error: {str(e)}")
                return f"Error generating detailed report: {str(e)}"
        
        return Tool(
            name="generate_detailed_report",
            func=generate_detailed_report,
            description="Generate comprehensive detailed data quality report"
        )
    
    def generate_visualizations_tool(self) -> Tool:
        """Tool to generate visualizations"""
        
        def generate_visualizations(table_name: str, visualization_type: str = "all") -> str:
            """Generate data quality visualizations"""
            try:
                # Get data
                schema_df = self.snowflake_manager.get_table_schema(table_name)
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=5000)
                
                if sample_df.empty:
                    return f"No data found in table: {table_name}"
                
                visualizations = []
                
                # 1. Null value heatmap
                if visualization_type in ["all", "nulls"]:
                    null_matrix = sample_df.isnull().astype(int)
                    fig = px.imshow(
                        null_matrix.T,
                        title=f"Null Value Distribution - {table_name}",
                        labels=dict(x="Row Index", y="Column", color="Is Null"),
                        color_continuous_scale="Reds"
                    )
                    fig.update_layout(height=400)
                    
                    # Convert to HTML for display
                    null_viz = fig.to_html(full_html=False, include_plotlyjs='cdn')
                    visualizations.append(("Null Value Heatmap", null_viz))
                
                # 2. Data type distribution
                if visualization_type in ["all", "types"] and not schema_df.empty:
                    type_counts = schema_df['DATA_TYPE'].value_counts()
                    fig = px.pie(
                        values=type_counts.values,
                        names=type_counts.index,
                        title=f"Data Type Distribution - {table_name}",
                        hole=0.3
                    )
                    fig.update_layout(height=400)
                    
                    type_viz = fig.to_html(full_html=False, include_plotlyjs='cdn')
                    visualizations.append(("Data Type Distribution", type_viz))
                
                # 3. Numeric column distributions
                if visualization_type in ["all", "numeric"]:
                    numeric_cols = [
                        col for col in sample_df.columns 
                        if pd.api.types.is_numeric_dtype(sample_df[col])
                    ]
                    
                    if numeric_cols:
                        # Create subplots for first 4 numeric columns
                        n_cols = min(4, len(numeric_cols))
                        fig = make_subplots(
                            rows=2, 
                            cols=2,
                            subplot_titles=numeric_cols[:n_cols]
                        )
                        
                        for idx, col in enumerate(numeric_cols[:n_cols]):
                            row = (idx // 2) + 1
                            col_pos = (idx % 2) + 1
                            
                            fig.add_trace(
                                go.Histogram(x=sample_df[col].dropna(), name=col),
                                row=row, col=col_pos
                            )
                        
                        fig.update_layout(
                            height=600,
                            title_text=f"Numeric Column Distributions - {table_name}",
                            showlegend=False
                        )
                        
                        numeric_viz = fig.to_html(full_html=False, include_plotlyjs='cdn')
                        visualizations.append(("Numeric Distributions", numeric_viz))
                
                if not visualizations:
                    return "No visualizations generated. Check data availability."
                
                # Format output
                output = "## 📊 Generated Visualizations\n\n"
                for viz_name, viz_html in visualizations:
                    output += f"### {viz_name}\n\n"
                    output += f"{viz_html}\n\n"
                    output += "---\n\n"
                
                return output
                
            except Exception as e:
                logger.error(f"Visualization generation error: {str(e)}")
                return f"Error generating visualizations: {str(e)}"
        
        return Tool(
            name="generate_visualizations",
            func=generate_visualizations,
            description="Generate data quality visualizations (null heatmaps, type distributions, histograms)"
        )
    
    def export_report_tool(self) -> Tool:
        """Tool to export reports in various formats"""
        
        def export_report(table_name: str, format_type: str = "html") -> str:
            """Export data quality report in specified format"""
            try:
                # Get data for report
                row_count = self.snowflake_manager.get_row_count(table_name)
                schema_df = self.snowflake_manager.get_table_schema(table_name)
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=5000)
                
                if sample_df.empty:
                    return f"No data found in table: {table_name}"
                
                # Analyze data
                analysis = self.data_analyzer.comprehensive_analysis(sample_df, schema_df)
                
                if format_type.lower() == "html":
                    # Generate HTML report
                    html_report = self._generate_html_report(table_name, analysis, row_count)
                    return f"HTML report generated for {table_name}. Length: {len(html_report)} characters."
                
                elif format_type.lower() == "markdown":
                    # Generate Markdown report
                    md_report = self._generate_markdown_report(table_name, analysis, row_count)
                    return f"Markdown report generated for {table_name}. Length: {len(md_report)} characters."
                
                elif format_type.lower() == "json":
                    # Generate JSON report
                    import json
                    json_report = json.dumps(analysis, indent=2, default=str)
                    return f"JSON report generated for {table_name}. Length: {len(json_report)} characters."
                
                else:
                    return f"Unsupported format: {format_type}. Supported formats: html, markdown, json"
                
            except Exception as e:
                logger.error(f"Report export error: {str(e)}")
                return f"Error exporting report: {str(e)}"
        
        return Tool(
            name="export_report",
            func=export_report,
            description="Export data quality report in various formats (HTML, Markdown, JSON)"
        )
    
    def create_action_plan_tool(self) -> Tool:
        """Tool to create actionable improvement plan"""
        
        def create_action_plan(table_name: str, priority: str = "high") -> str:
            """Create actionable improvement plan based on findings"""
            try:
                # Get analysis
                schema_df = self.snowflake_manager.get_table_schema(table_name)
                sample_df = self.snowflake_manager.get_sample_data(table_name, limit=5000)
                
                if sample_df.empty:
                    return f"No data found in table: {table_name}"
                
                analysis = self.data_analyzer.comprehensive_analysis(sample_df, schema_df)
                issues = analysis.get('data_quality_issues', [])
                
                # Categorize issues by priority
                critical_issues = [issue for issue in issues if issue.get('severity') == 'critical']
                warning_issues = [issue for issue in issues if issue.get('severity') == 'warning']
                info_issues = [issue for issue in issues if issue.get('severity') == 'info']
                
                # Generate action plan
                action_plan = f"""
                # 🎯 Data Quality Improvement Plan
                
                **Table:** {table_name}
                **Generated:** {datetime.now().strftime('%Y-%m-%d')}
                **Priority Focus:** {priority.upper()}
                
                ## 📊 Current State Assessment
                - **Quality Score:** {analysis.get('data_quality_score', 0):.1f}/100
                - **Total Issues:** {len(issues)}
                - **Critical Issues:** {len(critical_issues)}
                - **Warning Issues:** {len(warning_issues)}
                
                ## 🚨 Immediate Actions (Week 1)
                """
                
                # Add immediate actions for critical issues
                if critical_issues:
                    for i, issue in enumerate(critical_issues[:3], 1):
                        action_plan += f"\n{i}. **{issue['column']}** - {issue['issue_type'].replace('_', ' ').title()}"
                        action_plan += f"\n   - **Action:** {self._get_remediation_action(issue)}"
                        action_plan += f"\n   - **Owner:** Data Steward"
                        action_plan += f"\n   - **Due:** Immediate"
                else:
                    action_plan += "\nNo critical issues requiring immediate action."
                
                action_plan += "\n\n## 📋 Short-term Improvements (Month 1)"
                
                # Add short-term actions
                all_issues = critical_issues + warning_issues
                if all_issues:
                    for i, issue in enumerate(all_issues[:5], 1):
                        action_plan += f"\n{i}. **{issue['column']}** - {issue['issue_type'].replace('_', ' ').title()}"
                        action_plan += f"\n   - **Action:** {self._get_remediation_action(issue)}"
                        action_plan += f"\n   - **Owner:** Data Analyst"
                        action_plan += f"\n   - **Due:** {datetime.now().strftime('%Y-%m')}"
                else:
                    action_plan += "\nNo issues requiring short-term action."
                
                action_plan += "\n\n## 🏗️ Long-term Initiatives (Quarter 1)"
                
                # Add long-term initiatives
                action_plan += """
                1. **Implement Data Validation Framework**
                   - Develop comprehensive validation rules
                   - Set up automated data quality checks
                   - Create alerting system
                
                2. **Establish Data Governance**
                   - Define data ownership and stewardship
                   - Create data quality SLAs
                   - Implement data catalog
                
                3. **Enhance Monitoring & Reporting**
                   - Set up dashboard for data quality metrics
                   - Create regular reporting cadence
                   - Implement trend analysis
                """
                
                action_plan += "\n\n## 📈 Success Metrics"
                action_plan += "\n- Reduce critical issues by 90% within 30 days"
                action_plan += "\n- Improve overall quality score to 95+ within 90 days"
                action_plan += "\n- Establish automated monitoring for 100% of critical columns"
                
                return action_plan
                
            except Exception as e:
                logger.error(f"Action plan creation error: {str(e)}")
                return f"Error creating action plan: {str(e)}"
        
        return Tool(
            name="create_action_plan",
            func=create_action_plan,
            description="Create actionable improvement plan with timelines and owners"
        )
    
    def _generate_html_report(self, table_name: str, analysis: Dict[str, Any], row_count: int) -> str:
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report - {table_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .metric {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .critical {{ color: #e74c3c; font-weight: bold; }}
                .warning {{ color: #f39c12; }}
                .info {{ color: #3498db; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>📊 Data Quality Report</h1>
            <h2>{table_name}</h2>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric">
                <h3>Key Metrics</h3>
                <p><strong>Quality Score:</strong> {analysis.get('data_quality_score', 0):.1f}/100</p>
                <p><strong>Total Rows:</strong> {row_count:,}</p>
                <p><strong>Total Issues:</strong> {len(analysis.get('data_quality_issues', []))}</p>
            </div>
        """
        
        # Add issues table
        issues = analysis.get('data_quality_issues', [])
        if issues:
            html += "<h2>Data Quality Issues</h2>"
            html += "<table>"
            html += "<tr><th>Column</th><th>Issue Type</th><th>Severity</th><th>Description</th></tr>"
            
            for issue in issues[:20]:  # Show first 20 issues
                severity_class = issue.get('severity', 'info')
                html += f"""
                <tr>
                    <td>{issue.get('column', 'N/A')}</td>
                    <td>{issue.get('issue_type', 'N/A').replace('_', ' ').title()}</td>
                    <td class="{severity_class}">{severity_class.upper()}</td>
                    <td>{issue.get('description', 'N/A')}</td>
                </tr>
                """
            
            html += "</table>"
            
            if len(issues) > 20:
                html += f"<p>... and {len(issues) - 20} more issues</p>"
        
        # Add recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            html += "<h2>Recommendations</h2><ul>"
            for rec in recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_markdown_report(self, table_name: str, analysis: Dict[str, Any], row_count: int) -> str:
        """Generate Markdown report"""
        md = f"""# Data Quality Report - {table_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Metrics
- **Quality Score:** {analysis.get('data_quality_score', 0):.1f}/100
- **Total Rows:** {row_count:,}
- **Total Columns:** {analysis.get('column_count', 0)}
- **Total Issues:** {len(analysis.get('data_quality_issues', []))}

## Data Quality Issues
"""
        
        issues = analysis.get('data_quality_issues', [])
        if issues:
            for issue in issues[:10]:  # Show first 10 issues
                severity = issue.get('severity', 'info').upper()
                md += f"\n### {severity} - {issue.get('column', 'N/A')}"
                md += f"\n- **Type:** {issue.get('issue_type', 'N/A').replace('_', ' ').title()}"
                md += f"\n- **Description:** {issue.get('description', 'N/A')}"
                md += f"\n- **Count:** {issue.get('count', 'N/A')}"
            
            if len(issues) > 10:
                md += f"\n\n... and {len(issues) - 10} more issues"
        
        # Add recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            md += "\n\n## Recommendations\n"
            for rec in recommendations:
                md += f"\n- {rec}"
        
        return md
    
    def _get_remediation_action(self, issue: Dict[str, Any]) -> str:
        """Get remediation action for an issue"""
        issue_type = issue.get('issue_type', '')
        
        if issue_type == 'null_values':
            return "Implement NOT NULL constraint or default value"
        elif issue_type == 'data_type_mismatch':
            return "Correct data type and validate conversion"
        elif issue_type == 'duplicate_rows':
            return "Add unique constraint and clean existing duplicates"
        elif issue_type == 'skewed_distribution':
            return "Investigate data collection process"
        elif issue_type == 'constant_value':
            return "Verify if column is needed or contains valid data"
        else:
            return "Review data quality and implement validation"
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reporting workflow"""
        try:
            logger.info(f"Starting report generation for table: {input_data.get('table_name')}")
            
            if not self.validate_input(input_data):
                return {
                    'success': False,
                    'error': 'Invalid input data'
                }
            
            table_name = input_data['table_name']
            
            # Use agent executor if we have tools
            if self.agent_executor:
                # Prepare input for agent
                agent_input = {
                    'input': f"Generate comprehensive report for table: {table_name}"
                }
                
                # Execute agent
                result = await self.agent_executor.ainvoke(agent_input)
                
                return {
                    'success': True,
                    'table_name': table_name,
                    'report': result['output'],
                    'agent_used': True,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Fallback to direct reporting
                return await self._direct_report_generation(table_name)
                
        except Exception as e:
            logger.error(f"Reporting agent error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _direct_report_generation(self, table_name: str) -> Dict[str, Any]:
        """Direct report generation without agent tools"""
        try:
            # Get data
            row_count = self.snowflake_manager.get_row_count(table_name)
            schema_df = self.snowflake_manager.get_table_schema(table_name)
            sample_df = self.snowflake_manager.get_sample_data(table_name, limit=5000)
            
            if sample_df.empty:
                return {
                    'success': False,
                    'error': f"No data found in table: {table_name}"
                }
            
            # Generate comprehensive analysis
            analysis = self.data_analyzer.comprehensive_analysis(sample_df, schema_df)
            
            # Generate report sections
            report_sections = {
                'executive_summary': self._generate_executive_summary(table_name, analysis, row_count),
                'detailed_findings': self._generate_detailed_findings(analysis),
                'recommendations': self._generate_recommendations_section(analysis),
                'visualizations': self._generate_visualization_data(sample_df, schema_df)
            }
            
            return {
                'success': True,
                'table_name': table_name,
                'analysis': analysis,
                'report_sections': report_sections,
                'full_report': self._combine_report_sections(report_sections),
                'agent_used': False,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Direct report generation error: {str(e)}")
            raise
    
    def _generate_executive_summary(self, table_name: str, analysis: Dict[str, Any], row_count: int) -> str:
        """Generate executive summary"""
        quality_score = analysis.get('data_quality_score', 0)
        total_issues = len(analysis.get('data_quality_issues', []))
        critical_issues = sum(1 for issue in analysis.get('data_quality_issues', []) 
                             if issue.get('severity') == 'critical')
        
        summary = f"""
        # Executive Summary - {table_name}
        
        ## 📊 Overview
        - **Table:** {table_name}
        - **Total Rows:** {row_count:,}
        - **Data Quality Score:** {quality_score:.1f}/100
        - **Status:** {'🟢 Good' if quality_score >= 80 else '🟡 Needs Attention' if quality_score >= 60 else '🔴 Poor'}
        
        ## 🚨 Key Findings
        - **Total Issues:** {total_issues}
        - **Critical Issues:** {critical_issues}
        - **Warning Issues:** {total_issues - critical_issues}
        
        ## 🎯 Top Priority
        """
        
        # Add top critical issues
        critical_issues = [issue for issue in analysis.get('data_quality_issues', []) 
                          if issue.get('severity') == 'critical']
        
        if critical_issues:
            for i, issue in enumerate(critical_issues[:3], 1):
                summary += f"\n{i}. **{issue.get('column', 'N/A')}**: {issue.get('description', 'N/A')}"
        else:
            summary += "\nNo critical issues identified"
        
        return summary
    
    def _generate_detailed_findings(self, analysis: Dict[str, Any]) -> str:
        """Generate detailed findings section"""
        issues = analysis.get('data_quality_issues', [])
        
        if not issues:
            return "## Detailed Findings\n\nNo data quality issues detected."
        
        # Group by issue type
        issues_by_type = {}
        for issue in issues:
            issue_type = issue.get('issue_type', 'other')
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        findings = "## Detailed Findings\n\n"
        
        for issue_type, type_issues in issues_by_type.items():
            findings += f"\n### {issue_type.replace('_', ' ').title()} ({len(type_issues)} issues)\n"
            
            for issue in type_issues[:5]:  # Show top 5 of each type
                severity = issue.get('severity', 'info').upper()
                findings += f"\n- **{severity}**: {issue.get('column', 'N/A')} - {issue.get('description', 'N/A')}"
            
            if len(type_issues) > 5:
                findings += f"\n- ... and {len(type_issues) - 5} more"
        
        return findings
    
    def _generate_recommendations_section(self, analysis: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        recommendations = analysis.get('recommendations', [])
        
        if not recommendations:
            recommendations = [
                "Maintain current data quality monitoring",
                "Continue regular data validation checks",
                "Review data entry processes periodically"
            ]
        
        section = "## Recommendations\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            section += f"{i}. {rec}\n"
        
        # Add timeline suggestions
        section += "\n### Suggested Timeline\n"
        section += "- **Immediate (1 week):** Address critical issues\n"
        section += "- **Short-term (1 month):** Implement validation rules\n"
        section += "- **Long-term (3 months):** Establish data governance\n"
        
        return section
    
    def _generate_visualization_data(self, df: pd.DataFrame, schema_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate visualization data"""
        try:
            viz_data = {}
            
            # Null percentages
            null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
            viz_data['null_percentages'] = {
                'columns': null_pct.head(10).index.tolist(),
                'percentages': null_pct.head(10).values.tolist()
            }
            
            # Data type distribution
            if not schema_df.empty:
                type_counts = schema_df['DATA_TYPE'].value_counts()
                viz_data['data_types'] = {
                    'types': type_counts.index.tolist(),
                    'counts': type_counts.values.tolist()
                }
            
            return viz_data
            
        except Exception as e:
            logger.error(f"Visualization data error: {str(e)}")
            return {}
    
    def _combine_report_sections(self, sections: Dict[str, Any]) -> str:
        """Combine all report sections"""
        report = sections.get('executive_summary', '') + "\n\n"
        report += sections.get('detailed_findings', '') + "\n\n"
        report += sections.get('recommendations', '') + "\n\n"
        
        # Add footer
        report += f"\n---\n\n*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        return report
    
    def get_required_input_fields(self) -> list:
        """Get required input fields"""
        return ['table_name']