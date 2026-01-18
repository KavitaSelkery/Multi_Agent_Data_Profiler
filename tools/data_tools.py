"""
Data analysis and manipulation tools
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

class DataAnalyzer:
    """Tools for data analysis and quality checking"""
    
    def __init__(self):
        """Initialize data analyzer"""
        pass
    
    def analyze_data_quality(self, df: pd.DataFrame, schema_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze data quality issues"""
        issues = []
        
        try:
            # Check for null values
            null_analysis = self._analyze_nulls(df, schema_df)
            issues.extend(null_analysis)
            
            # Check data types
            type_analysis = self._analyze_data_types(df, schema_df)
            issues.extend(type_analysis)
            
            # Check for duplicates
            duplicate_analysis = self._analyze_duplicates(df)
            issues.extend(duplicate_analysis)
            
            # Check value distributions
            distribution_analysis = self._analyze_distributions(df)
            issues.extend(distribution_analysis)
            
            return issues
            
        except Exception as e:
            logger.error(f"Data quality analysis error: {str(e)}")
            return []
    
    def _analyze_nulls(self, df: pd.DataFrame, schema_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze null values"""
        issues = []
        
        for column in df.columns:
            null_count = df[column].isnull().sum()
            total_count = len(df)
            
            if null_count > 0:
                null_percentage = (null_count / total_count) * 100
                
                # Check if column should not be null
                should_not_be_null = False
                if not schema_df.empty and column in schema_df['COLUMN_NAME'].values:
                    col_info = schema_df[schema_df['COLUMN_NAME'] == column].iloc[0]
                    should_not_be_null = col_info['IS_NULLABLE'].upper() == 'NO'
                
                severity = 'critical' if should_not_be_null and null_count > 0 else 'warning'
                
                issues.append({
                    'column': column,
                    'issue_type': 'null_values',
                    'count': int(null_count),
                    'percentage': float(null_percentage),
                    'severity': severity,
                    'description': f'Found {null_count} null values ({null_percentage:.1f}%)'
                })
        
        return issues
    
    def _analyze_data_types(self, df: pd.DataFrame, schema_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze data type issues"""
        issues = []
        
        if schema_df.empty:
            return issues
        
        for _, col_info in schema_df.iterrows():
            column_name = col_info['COLUMN_NAME']
            expected_type = col_info['DATA_TYPE']
            
            if column_name not in df.columns:
                continue
            
            # Get actual dtype
            actual_dtype = str(df[column_name].dtype)
            
            # Check for type mismatches
            if expected_type.upper() != actual_dtype.upper():
                # Check if it's a critical mismatch
                if self._is_numeric_type(expected_type) and not self._is_numeric_type(actual_dtype):
                    # Check if string values can be converted to numeric
                    can_convert, conversion_rate = self._check_numeric_conversion(df[column_name])
                    
                    if can_convert:
                        severity = 'warning'
                        desc = f'Type mismatch but {conversion_rate:.1f}% convertible'
                    else:
                        severity = 'critical'
                        desc = f'Critical type mismatch: expected numeric, got {actual_dtype}'
                else:
                    severity = 'info'
                    desc = f'Type mismatch: expected {expected_type}, got {actual_dtype}'
                
                issues.append({
                    'column': column_name,
                    'issue_type': 'data_type_mismatch',
                    'expected_type': expected_type,
                    'actual_type': actual_dtype,
                    'severity': severity,
                    'description': desc
                })
        
        return issues
    
    def _is_numeric_type(self, dtype: str) -> bool:
        """Check if data type is numeric"""
        numeric_types = ['NUMBER', 'INT', 'INTEGER', 'FLOAT', 'DECIMAL', 'DOUBLE', 'NUMERIC']
        return dtype.upper() in numeric_types
    
    def _check_numeric_conversion(self, series: pd.Series) -> Tuple[bool, float]:
        """Check if string series can be converted to numeric"""
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            non_null_count = numeric_series.notnull().sum()
            total_count = len(series)
            
            conversion_rate = (non_null_count / total_count) * 100 if total_count > 0 else 0
            can_convert = conversion_rate >= 95
            
            return can_convert, conversion_rate
            
        except:
            return False, 0
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze duplicate rows"""
        issues = []
        
        try:
            duplicate_count = df.duplicated().sum()
            
            if duplicate_count > 0:
                duplicate_percentage = (duplicate_count / len(df)) * 100
                
                issues.append({
                    'issue_type': 'duplicate_rows',
                    'count': int(duplicate_count),
                    'percentage': float(duplicate_percentage),
                    'severity': 'warning',
                    'description': f'Found {duplicate_count} duplicate rows ({duplicate_percentage:.1f}%)'
                })
            
            return issues
            
        except Exception as e:
            logger.error(f"Duplicate analysis error: {str(e)}")
            return []
    
    def _analyze_distributions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze value distributions"""
        issues = []
        
        for column in df.columns:
            # Skip non-numeric columns for distribution analysis
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
            
            # Check for skewed distributions
            skewness = df[column].skew()
            
            if abs(skewness) > 2:
                issues.append({
                    'column': column,
                    'issue_type': 'skewed_distribution',
                    'skewness': float(skewness),
                    'severity': 'info',
                    'description': f'Highly skewed distribution (skewness: {skewness:.2f})'
                })
            
            # Check for constant values
            unique_count = df[column].nunique()
            if unique_count == 1 and len(df) > 10:
                issues.append({
                    'column': column,
                    'issue_type': 'constant_value',
                    'value': df[column].iloc[0],
                    'severity': 'warning',
                    'description': f'Constant value "{df[column].iloc[0]}" for all rows'
                })
        
        return issues
    
    def detect_anomalies(self, df: pd.DataFrame, method: str = 'IQR') -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        anomalies = []
        
        try:
            for column in df.columns:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    continue
                
                # Clean series
                series = df[column].dropna()
                if len(series) < 10:
                    continue
                
                if method == 'IQR':
                    column_anomalies = self._detect_iqr_anomalies(series, column)
                elif method == 'Z-Score':
                    column_anomalies = self._detect_zscore_anomalies(series, column)
                elif method == 'Isolation Forest':
                    column_anomalies = self._detect_isolation_forest_anomalies(series, column)
                else:
                    column_anomalies = self._detect_iqr_anomalies(series, column)
                
                anomalies.extend(column_anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")
            return []
    
    def _detect_iqr_anomalies(self, series: pd.Series, column_name: str) -> List[Dict[str, Any]]:
        """Detect anomalies using IQR method"""
        anomalies = []
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        if not outliers.empty:
            anomalies.append({
                'column': column_name,
                'anomaly_type': 'iqr_outlier',
                'count': len(outliers),
                'percentage': (len(outliers) / len(series)) * 100,
                'min_value': float(outliers.min()),
                'max_value': float(outliers.max()),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'severity': 'warning' if len(outliers) < len(series) * 0.05 else 'critical'
            })
        
        return anomalies
    
    def _detect_zscore_anomalies(self, series: pd.Series, column_name: str) -> List[Dict[str, Any]]:
        """Detect anomalies using Z-Score method"""
        anomalies = []
        
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return anomalies
        
        z_scores = (series - mean) / std
        outliers = series[abs(z_scores) > 3]  # 3 standard deviations
        
        if not outliers.empty:
            anomalies.append({
                'column': column_name,
                'anomaly_type': 'zscore_outlier',
                'count': len(outliers),
                'percentage': (len(outliers) / len(series)) * 100,
                'min_zscore': float(z_scores.min()),
                'max_zscore': float(z_scores.max()),
                'severity': 'warning'
            })
        
        return anomalies
    
    def _detect_isolation_forest_anomalies(self, series: pd.Series, column_name: str) -> List[Dict[str, Any]]:
        """Detect anomalies using Isolation Forest"""
        anomalies = []
        
        try:
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            predictions = iso_forest.fit_predict(X)
            
            # Identify anomalies (predictions == -1)
            outlier_indices = np.where(predictions == -1)[0]
            
            if len(outlier_indices) > 0:
                outlier_values = series.iloc[outlier_indices]
                
                anomalies.append({
                    'column': column_name,
                    'anomaly_type': 'isolation_forest_outlier',
                    'count': len(outlier_values),
                    'percentage': (len(outlier_values) / len(series)) * 100,
                    'min_value': float(outlier_values.min()),
                    'max_value': float(outlier_values.max()),
                    'severity': 'warning'
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Isolation Forest error: {str(e)}")
            return []
    
    def comprehensive_analysis(self, df: pd.DataFrame, schema_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data analysis"""
        try:
            analysis = {
                'row_count': len(df),
                'column_count': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'data_quality_issues': self.analyze_data_quality(df, schema_df),
                'statistical_summary': self._generate_statistical_summary(df),
                'recommendations': []
            }
            
            # Generate recommendations based on findings
            recommendations = self._generate_recommendations(analysis['data_quality_issues'])
            analysis['recommendations'] = recommendations
            
            # Calculate data quality score
            analysis['data_quality_score'] = self._calculate_quality_score(analysis['data_quality_issues'])
            
            return analysis
            
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {str(e)}")
            return {}
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary"""
        summary = {}
        
        for column in df.columns:
            col_summary = {
                'dtype': str(df[column].dtype),
                'null_count': int(df[column].isnull().sum()),
                'unique_count': int(df[column].nunique())
            }
            
            if pd.api.types.is_numeric_dtype(df[column]):
                col_summary.update({
                    'min': float(df[column].min()),
                    'max': float(df[column].max()),
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std()),
                    'median': float(df[column].median())
                })
            
            summary[column] = col_summary
        
        return summary
    
    def _generate_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on issues"""
        recommendations = []
        
        critical_issues = [issue for issue in issues if issue.get('severity') == 'critical']
        warning_issues = [issue for issue in issues if issue.get('severity') == 'warning']
        
        if critical_issues:
            recommendations.append("Address critical data quality issues immediately")
        
        if warning_issues:
            recommendations.append("Review and fix warning-level issues")
        
        # Specific recommendations
        for issue in issues:
            if issue['issue_type'] == 'null_values' and issue['percentage'] > 50:
                recommendations.append(f"Consider removing or imputing values in {issue['column']} (high null percentage)")
            
            if issue['issue_type'] == 'data_type_mismatch' and issue['severity'] == 'critical':
                recommendations.append(f"Fix data type for {issue['column']} - expected {issue['expected_type']}")
        
        return recommendations
    
    def _calculate_quality_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate data quality score (0-100)"""
        if not issues:
            return 100.0
        
        score = 100.0
        
        for issue in issues:
            if issue['severity'] == 'critical':
                score -= 5
            elif issue['severity'] == 'warning':
                score -= 2
            elif issue['severity'] == 'info':
                score -= 0.5
        
        return max(0.0, min(100.0, score))