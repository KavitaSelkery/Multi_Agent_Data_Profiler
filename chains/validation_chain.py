"""
LangChain chains for data validation
"""
from typing import Dict, Any, List, Optional
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from loguru import logger

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import Settings

class ValidationResult(BaseModel):
    """Pydantic model for validation output"""
    is_valid: bool = Field(description="Whether the data is valid")
    issues_found: List[str] = Field(description="List of validation issues")
    severity: str = Field(description="Overall severity (critical, warning, info)")
    recommendations: List[str] = Field(description="Recommendations for fixing issues")
    validation_score: float = Field(description="Validation score (0-100)")

class SchemaValidationResult(BaseModel):
    """Pydantic model for schema validation output"""
    schema_compliance: float = Field(description="Schema compliance percentage")
    missing_columns: List[str] = Field(description="Columns missing from data")
    type_mismatches: Dict[str, str] = Field(description="Data type mismatches")
    constraint_violations: List[str] = Field(description="Constraint violations")

class DataValidationChain:
    """Chain for data validation against rules and schemas"""
    
    def __init__(self):
        """Initialize validation chain"""
        self.settings = Settings()
        self.llm = ChatOpenAI(
            model=self.settings.OPENAI_MODEL,
            temperature=0.1,
            api_key=self.settings.OPENAI_API_KEY
        )
        self.validation_parser = PydanticOutputParser(pydantic_object=ValidationResult)
        self.schema_parser = PydanticOutputParser(pydantic_object=SchemaValidationResult)
        
        self._initialize_chains()
    
    def _initialize_chains(self):
        """Initialize all chains"""
        # General validation chain
        self.validation_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Validate the following data against these rules:\n\n"
                "Data Summary:\n{data_summary}\n\n"
                "Validation Rules:\n{rules}\n\n"
                "Provide validation results with:\n"
                "1. Overall validity\n2. Issues found\n3. Severity level\n4. Recommendations\n5. Validation score\n\n"
                "{format_instructions}",
                input_variables=["data_summary", "rules"],
                partial_variables={"format_instructions": self.validation_parser.get_format_instructions()}
            ),
            output_key="validation_result"
        )
        
        # Schema validation chain
        self.schema_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Validate data schema compliance:\n\n"
                "Actual Schema:\n{actual_schema}\n\n"
                "Expected Schema:\n{expected_schema}\n\n"
                "Data Sample:\n{data_sample}\n\n"
                "Check for:\n1. Missing columns\n2. Data type mismatches\n3. Constraint violations\n4. Schema compliance percentage\n\n"
                "{format_instructions}",
                input_variables=["actual_schema", "expected_schema", "data_sample"],
                partial_variables={"format_instructions": self.schema_parser.get_format_instructions()}
            ),
            output_key="schema_result"
        )
        
        # Business rule validation chain
        self.business_rule_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Validate data against business rules:\n\n"
                "Data Context:\n{data_context}\n\n"
                "Business Rules:\n{rules}\n\n"
                "Data Issues Found:\n{issues}\n\n"
                "Assess business impact and provide recommendations.",
                input_variables=["data_context", "rules", "issues"]
            ),
            output_key="business_validation"
        )
        
        # Constraint checking chain
        self.constraint_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Check data constraints:\n\n"
                "Data Summary:\n{data_summary}\n\n"
                "Constraints:\n{constraints}\n\n"
                "Identify constraint violations and suggest fixes.",
                input_variables=["data_summary", "constraints"]
            ),
            output_key="constraint_check"
        )
    
    async def validate_data(self, data_summary: Dict[str, Any], rules: List[str]) -> Dict[str, Any]:
        """Validate data against rules"""
        try:
            # Format data summary
            formatted_summary = self._format_data_summary(data_summary)
            
            # Format rules
            formatted_rules = "\n".join([f"- {rule}" for rule in rules])
            
            # Run validation
            validation_result = await self.validation_chain.arun(
                data_summary=formatted_summary,
                rules=formatted_rules
            )
            
            # Parse result
            try:
                parsed_result = self.validation_parser.parse(validation_result)
                result_dict = parsed_result.dict()
            except Exception as parse_error:
                logger.warning(f"Parse error: {parse_error}")
                result_dict = {
                    'is_valid': False,
                    'issues_found': ['Validation parsing failed'],
                    'severity': 'critical',
                    'recommendations': ['Check validation rules'],
                    'validation_score': 0
                }
            
            return {
                'success': True,
                'validation_result': result_dict,
                'raw_data_summary': data_summary,
                'rules_applied': rules
            }
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_data_summary(self, data_summary: Dict[str, Any]) -> str:
        """Format data summary for LLM"""
        formatted = []
        
        if 'row_count' in data_summary:
            formatted.append(f"Row Count: {data_summary['row_count']:,}")
        
        if 'column_count' in data_summary:
            formatted.append(f"Column Count: {data_summary['column_count']}")
        
        if 'issues' in data_summary and data_summary['issues']:
            formatted.append("\nData Quality Issues:")
            for issue in data_summary['issues'][:10]:
                severity = issue.get('severity', 'unknown').upper()
                formatted.append(f"- [{severity}] {issue.get('description', 'No description')}")
        
        if 'statistics' in data_summary:
            formatted.append("\nKey Statistics:")
            for col, stats in list(data_summary['statistics'].items())[:5]:
                formatted.append(f"\n{col}:")
                if 'null_count' in stats:
                    formatted.append(f"  - Nulls: {stats['null_count']}")
                if 'unique_count' in stats:
                    formatted.append(f"  - Unique: {stats['unique_count']}")
                if 'min' in stats and 'max' in stats:
                    formatted.append(f"  - Range: {stats['min']} to {stats['max']}")
        
        return "\n".join(formatted)
    
    async def validate_schema(self, actual_schema: Dict[str, Any], 
                            expected_schema: Dict[str, Any], 
                            data_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schema compliance"""
        try:
            # Format schemas
            formatted_actual = self._format_schema(actual_schema)
            formatted_expected = self._format_schema(expected_schema)
            formatted_sample = self._format_data_sample(data_sample)
            
            # Run schema validation
            schema_result = await self.schema_chain.arun(
                actual_schema=formatted_actual,
                expected_schema=formatted_expected,
                data_sample=formatted_sample
            )
            
            # Parse result
            try:
                parsed_result = self.schema_parser.parse(schema_result)
                result_dict = parsed_result.dict()
            except Exception as parse_error:
                logger.warning(f"Schema parse error: {parse_error}")
                result_dict = {
                    'schema_compliance': 0,
                    'missing_columns': ['Schema parsing failed'],
                    'type_mismatches': {},
                    'constraint_violations': ['Check schema definitions']
                }
            
            # Calculate additional metrics
            total_columns = len(expected_schema.get('columns', []))
            if total_columns > 0:
                missing_count = len(result_dict.get('missing_columns', []))
                mismatch_count = len(result_dict.get('type_mismatches', {}))
                
                compliance_score = max(0, 100 - (
                    (missing_count / total_columns * 50) + 
                    (mismatch_count / total_columns * 30)
                ))
                result_dict['calculated_compliance'] = compliance_score
            
            return {
                'success': True,
                'schema_result': result_dict,
                'actual_schema': actual_schema,
                'expected_schema': expected_schema
            }
            
        except Exception as e:
            logger.error(f"Schema validation error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format schema for LLM"""
        formatted = []
        
        if 'table_name' in schema:
            formatted.append(f"Table: {schema['table_name']}")
        
        if 'columns' in schema and schema['columns']:
            formatted.append("\nColumns:")
            for col in schema['columns'][:20]:  # Limit to 20 columns
                col_info = f"- {col.get('name', 'Unknown')}: {col.get('type', 'Unknown')}"
                
                if col.get('should_not_be_null'):
                    col_info += " [NOT NULL]"
                if col.get('should_be_unique'):
                    col_info += " [UNIQUE]"
                if col.get('validation_rules'):
                    col_info += f" [Rules: {', '.join(col['validation_rules'])}]"
                
                formatted.append(col_info)
        
        return "\n".join(formatted)
    
    def _format_data_sample(self, data_sample: Dict[str, Any]) -> str:
        """Format data sample for LLM"""
        formatted = []
        
        if 'sample_rows' in data_sample and data_sample['sample_rows']:
            formatted.append("Sample Data (first 5 rows):")
            
            for i, row in enumerate(data_sample['sample_rows'][:5]):
                formatted.append(f"\nRow {i + 1}:")
                for key, value in list(row.items())[:5]:  # First 5 columns
                    value_str = str(value)[:50]  # Truncate long values
                    if len(str(value)) > 50:
                        value_str += "..."
                    formatted.append(f"  {key}: {value_str}")
        
        return "\n".join(formatted)
    
    async def validate_business_rules(self, data_context: Dict[str, Any], 
                                    rules: List[str], 
                                    issues: List[str]) -> Dict[str, Any]:
        """Validate against business rules"""
        try:
            # Format context
            formatted_context = self._format_business_context(data_context)
            formatted_rules = "\n".join([f"- {rule}" for rule in rules])
            formatted_issues = "\n".join([f"- {issue}" for issue in issues[:20]])
            
            # Run business rule validation
            business_result = await self.business_rule_chain.arun(
                data_context=formatted_context,
                rules=formatted_rules,
                issues=formatted_issues
            )
            
            # Extract key points
            impact_assessment = self._extract_business_impact(business_result)
            recommendations = self._extract_recommendations(business_result)
            
            return {
                'success': True,
                'business_validation': business_result,
                'impact_assessment': impact_assessment,
                'recommendations': recommendations,
                'rules_evaluated': rules
            }
            
        except Exception as e:
            logger.error(f"Business rule validation error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_business_context(self, context: Dict[str, Any]) -> str:
        """Format business context"""
        formatted = []
        
        if 'business_domain' in context:
            formatted.append(f"Business Domain: {context['business_domain']}")
        
        if 'critical_columns' in context:
            formatted.append(f"Critical Columns: {', '.join(context['critical_columns'])}")
        
        if 'data_sensitivity' in context:
            formatted.append(f"Data Sensitivity: {context['data_sensitivity']}")
        
        if 'compliance_requirements' in context:
            formatted.append(f"Compliance: {context['compliance_requirements']}")
        
        return "\n".join(formatted)
    
    def _extract_business_impact(self, validation_text: str) -> Dict[str, Any]:
        """Extract business impact assessment"""
        impact = {
            'financial': 'Unknown',
            'operational': 'Unknown',
            'compliance': 'Unknown',
            'reputation': 'Unknown'
        }
        
        lines = validation_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            if any(word in line_lower for word in ['financial', 'revenue', 'cost']):
                if 'high' in line_lower:
                    impact['financial'] = 'High'
                elif 'medium' in line_lower:
                    impact['financial'] = 'Medium'
                elif 'low' in line_lower:
                    impact['financial'] = 'Low'
            
            elif any(word in line_lower for word in ['operational', 'process', 'efficiency']):
                if 'high' in line_lower:
                    impact['operational'] = 'High'
                elif 'medium' in line_lower:
                    impact['operational'] = 'Medium'
                elif 'low' in line_lower:
                    impact['operational'] = 'Low'
            
            elif any(word in line_lower for word in ['compliance', 'regulatory', 'legal']):
                if 'high' in line_lower:
                    impact['compliance'] = 'High'
                elif 'medium' in line_lower:
                    impact['compliance'] = 'Medium'
                elif 'low' in line_lower:
                    impact['compliance'] = 'Low'
            
            elif any(word in line_lower for word in ['reputation', 'brand', 'trust']):
                if 'high' in line_lower:
                    impact['reputation'] = 'High'
                elif 'medium' in line_lower:
                    impact['reputation'] = 'Medium'
                elif 'low' in line_lower:
                    impact['reputation'] = 'Low'
        
        return impact
    
    def _extract_recommendations(self, validation_text: str) -> List[str]:
        """Extract recommendations from validation text"""
        recommendations = []
        lines = validation_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(word in line.lower() for word in ['recommend', 'suggest', 'should', 'must', 'need to']):
                # Clean up the line
                line = line.replace('-', '').replace('•', '').strip()
                if line and len(line) > 10:  # Meaningful length
                    recommendations.append(line)
        
        return recommendations[:10]  # Return top 10
    
    async def check_constraints(self, data_summary: Dict[str, Any], 
                              constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check data constraints"""
        try:
            # Format data summary
            formatted_summary = self._format_data_summary(data_summary)
            
            # Format constraints
            formatted_constraints = []
            for constraint in constraints:
                const_str = f"- {constraint.get('type', 'Unknown')}: "
                if 'column' in constraint:
                    const_str += f"Column '{constraint['column']}' "
                if 'condition' in constraint:
                    const_str += f"must satisfy: {constraint['condition']}"
                formatted_constraints.append(const_str)
            
            constraints_text = "\n".join(formatted_constraints)
            
            # Run constraint check
            constraint_result = await self.constraint_chain.arun(
                data_summary=formatted_summary,
                constraints=constraints_text
            )
            
            # Parse violations
            violations = self._extract_constraint_violations(constraint_result)
            
            return {
                'success': True,
                'constraint_check': constraint_result,
                'violations_found': violations,
                'constraints_checked': constraints
            }
            
        except Exception as e:
            logger.error(f"Constraint check error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_constraint_violations(self, check_text: str) -> List[Dict[str, Any]]:
        """Extract constraint violations from check text"""
        violations = []
        lines = check_text.split('\n')
        
        current_violation = {}
        
        for line in lines:
            line = line.strip()
            
            if any(word in line.lower() for word in ['violation', 'fails', 'breaks', 'invalid']):
                if current_violation:
                    violations.append(current_violation)
                
                current_violation = {
                    'constraint': line,
                    'severity': 'warning',
                    'suggestion': ''
                }
            
            elif 'severity' in line.lower():
                if 'critical' in line.lower():
                    current_violation['severity'] = 'critical'
                elif 'warning' in line.lower():
                    current_violation['severity'] = 'warning'
                elif 'info' in line.lower():
                    current_violation['severity'] = 'info'
            
            elif any(word in line.lower() for word in ['suggest', 'fix', 'correct']):
                current_violation['suggestion'] = line
        
        if current_violation:
            violations.append(current_violation)
        
        return violations