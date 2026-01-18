"""
LangChain chains for data profiling
"""
from typing import Dict, Any, List, Optional
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from loguru import logger
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import Settings
from config.prompts import PROFILER_AGENT_PROMPT

class ProfilingAnalysis(BaseModel):
    """Pydantic model for profiling analysis output"""
    summary: str = Field(description="Brief summary of findings")
    critical_issues: List[str] = Field(description="List of critical issues found")
    recommendations: List[str] = Field(description="Recommendations for improvement")
    quality_score: float = Field(description="Overall data quality score (0-100)")
    next_steps: List[str] = Field(description="Suggested next steps")

class DataProfilingChain:
    """Chain for data profiling analysis"""
    
    def __init__(self):
        """Initialize profiling chain"""
        self.settings = Settings()
        self.llm = ChatOpenAI(
            model=self.settings.OPENAI_MODEL,
            temperature=0.1,
            api_key=self.settings.OPENAI_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=ProfilingAnalysis)
        
        self._initialize_chains()
    
    def _initialize_chains(self):
        """Initialize all chains"""
        # Main profiling chain
        self.profiling_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PROFILER_AGENT_PROMPT + "\n\n" +
                "Analyze the following data profiling information:\n\n{profiling_data}\n\n" +
                "Provide a comprehensive analysis with the following structure:\n" +
                "1. Summary of findings\n" +
                "2. Critical issues\n" +
                "3. Recommendations\n" +
                "4. Quality score (0-100)\n" +
                "5. Next steps\n\n" +
                "{format_instructions}",
                input_variables=["profiling_data"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            ),
            output_key="analysis"
        )
        
        # Issue categorization chain
        self.categorization_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Categorize the following data quality issues by severity and type:\n\n{issues}\n\n" +
                "Format as:\n- Critical: [list]\n- Warning: [list]\n- Info: [list]\n",
                input_variables=["issues"]
            ),
            output_key="categorized_issues"
        )
        
        # Recommendation chain
        self.recommendation_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Based on these data quality issues, provide actionable recommendations:\n\n{issues}\n\n" +
                "Focus on:\n1. Immediate fixes\n2. Process improvements\n3. Preventive measures\n",
                input_variables=["issues"]
            ),
            output_key="recommendations"
        )
    
    async def analyze_profiling_data(self, profiling_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze profiling data using LLM"""
        try:
            # Format profiling data for LLM
            formatted_data = self._format_profiling_data(profiling_data)
            
            # Run main analysis chain
            analysis_result = await self.profiling_chain.arun(profiling_data=formatted_data)
            
            # Parse the result
            try:
                parsed_analysis = self.parser.parse(analysis_result)
                analysis_dict = parsed_analysis.dict()
            except:
                # Fallback if parsing fails
                analysis_dict = {
                    'summary': analysis_result,
                    'critical_issues': [],
                    'recommendations': [],
                    'quality_score': 0,
                    'next_steps': []
                }
            
            # Extract issues for categorization
            if 'data_quality_issues' in profiling_data:
                issues_text = "\n".join([
                    f"- {issue.get('column', 'Unknown')}: {issue.get('description', 'No description')}"
                    for issue in profiling_data['data_quality_issues'][:20]
                ])
                
                # Categorize issues
                categorized = await self.categorization_chain.arun(issues=issues_text)
                
                # Get specific recommendations
                recommendations = await self.recommendation_chain.arun(issues=issues_text)
                
                # Combine results
                analysis_dict['categorized_issues'] = categorized
                analysis_dict['detailed_recommendations'] = recommendations
            
            return {
                'success': True,
                'analysis': analysis_dict,
                'raw_profiling_data': profiling_data
            }
            
        except Exception as e:
            logger.error(f"Profiling analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_profiling_data(self, data: Dict[str, Any]) -> str:
        """Format profiling data for LLM consumption"""
        formatted = []
        
        # Basic info
        if 'row_count' in data:
            formatted.append(f"Total Rows: {data['row_count']:,}")
        if 'column_count' in data:
            formatted.append(f"Total Columns: {data['column_count']}")
        if 'memory_usage_mb' in data:
            formatted.append(f"Memory Usage: {data['memory_usage_mb']:.2f} MB")
        
        # Data quality issues
        if 'data_quality_issues' in data and data['data_quality_issues']:
            formatted.append("\n## Data Quality Issues")
            for issue in data['data_quality_issues'][:10]:  # Limit to 10 issues
                severity = issue.get('severity', 'unknown').upper()
                formatted.append(f"- [{severity}] {issue.get('column', 'Unknown')}: {issue.get('description', 'No description')}")
            
            if len(data['data_quality_issues']) > 10:
                formatted.append(f"- ... and {len(data['data_quality_issues']) - 10} more issues")
        
        # Statistical summary
        if 'statistical_summary' in data and data['statistical_summary']:
            formatted.append("\n## Statistical Summary")
            for column, stats in list(data['statistical_summary'].items())[:5]:  # First 5 columns
                formatted.append(f"\n### {column}")
                formatted.append(f"- Type: {stats.get('dtype', 'N/A')}")
                formatted.append(f"- Nulls: {stats.get('null_count', 0)}")
                formatted.append(f"- Unique: {stats.get('unique_count', 0)}")
                
                if 'min' in stats:
                    formatted.append(f"- Range: {stats.get('min'):.2f} to {stats.get('max'):.2f}")
                    formatted.append(f"- Mean: {stats.get('mean', 0):.2f}")
        
        return "\n".join(formatted)
    
    async def generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate executive summary from analysis"""
        try:
            prompt = PromptTemplate(
                template="Generate an executive summary (2-3 paragraphs) based on this data quality analysis:\n\n{analysis}\n\n" +
                "Focus on:\n1. Overall assessment\n2. Key findings\n3. Business impact\n4. Recommended actions",
                input_variables=["analysis"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            summary = await chain.arun(analysis=str(analysis))
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return "Unable to generate summary."
    
    async def suggest_improvements(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Suggest improvements for specific issues"""
        try:
            if not issues:
                return ["No issues to improve"]
            
            # Format issues
            issues_text = "\n".join([
                f"{i+1}. {issue.get('issue_type', 'Unknown')} in {issue.get('column', 'Unknown column')}: "
                f"{issue.get('description', 'No description')}"
                for i, issue in enumerate(issues[:10])
            ])
            
            prompt = PromptTemplate(
                template="For each of these data quality issues, suggest specific, actionable improvements:\n\n{issues}\n\n" +
                "Format each suggestion as:\n- Issue: [description]\n- Improvement: [actionable step]\n- Priority: [High/Medium/Low]",
                input_variables=["issues"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            improvements = await chain.arun(issues=issues_text)
            
            # Parse improvements into list
            return [line.strip() for line in improvements.split('\n') if line.strip()]
            
        except Exception as e:
            logger.error(f"Improvement suggestion error: {str(e)}")
            return ["Unable to generate improvement suggestions."]