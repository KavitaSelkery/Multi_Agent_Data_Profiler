"""
LangChain chains for anomaly detection
"""
from typing import Dict, Any, List
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from loguru import logger
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import Settings
from config.prompts import ANOMALY_AGENT_PROMPT

class AnomalyAnalysis(BaseModel):
    """Pydantic model for anomaly analysis output"""
    summary: str = Field(description="Brief summary of anomaly findings")
    anomaly_types: Dict[str, int] = Field(description="Count of anomalies by type")
    affected_columns: List[str] = Field(description="Columns with anomalies")
    severity_assessment: str = Field(description="Overall severity assessment")
    root_cause_analysis: List[str] = Field(description="Possible root causes")
    mitigation_strategies: List[str] = Field(description="Mitigation strategies")

class AnomalyDetectionChain:
    """Chain for anomaly detection analysis"""
    
    def __init__(self):
        """Initialize anomaly detection chain"""
        self.settings = Settings()
        self.llm = ChatOpenAI(
            model=self.settings.OPENAI_MODEL,
            temperature=0.1,
            api_key=self.settings.OPENAI_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=AnomalyAnalysis)
        
        self._initialize_chains()
    
    def _initialize_chains(self):
        """Initialize all chains"""
        # Main anomaly analysis chain
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=ANOMALY_AGENT_PROMPT + "\n\n" +
                "Analyze the following anomaly detection results:\n\n{anomaly_data}\n\n" +
                "Provide a comprehensive analysis with the following structure:\n" +
                "1. Summary of anomaly findings\n" +
                "2. Types and counts of anomalies\n" +
                "3. Affected columns\n" +
                "4. Severity assessment\n" +
                "5. Root cause analysis\n" +
                "6. Mitigation strategies\n\n" +
                "{format_instructions}",
                input_variables=["anomaly_data"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            ),
            output_key="analysis"
        )
        
        # Severity assessment chain
        self.severity_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Assess the severity of these anomalies on a scale of 1-10:\n\n{anomalies}\n\n" +
                "Consider:\n- Business impact\n- Data integrity risk\n- Frequency of occurrence\n- Ease of detection\n",
                input_variables=["anomalies"]
            ),
            output_key="severity_assessment"
        )
        
        # Root cause analysis chain
        self.root_cause_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Analyze potential root causes for these anomalies:\n\n{anomalies}\n\n" +
                "Consider:\n1. Data entry errors\n2. System issues\n3. Process failures\n4. External factors\n",
                input_variables=["anomalies"]
            ),
            output_key="root_causes"
        )
    
    async def analyze_anomalies(self, anomaly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze anomaly data using LLM"""
        try:
            # Format anomaly data for LLM
            formatted_data = self._format_anomaly_data(anomaly_data)
            
            # Run main analysis chain
            analysis_result = await self.analysis_chain.arun(anomaly_data=formatted_data)
            
            # Parse the result
            try:
                parsed_analysis = self.parser.parse(analysis_result)
                analysis_dict = parsed_analysis.dict()
            except Exception as parse_error:
                logger.warning(f"Parse error, using fallback: {parse_error}")
                analysis_dict = {
                    'summary': analysis_result,
                    'anomaly_types': {},
                    'affected_columns': [],
                    'severity_assessment': 'Unknown',
                    'root_cause_analysis': [],
                    'mitigation_strategies': []
                }
            
            # Additional severity assessment if anomalies exist
            if 'anomalies' in anomaly_data and anomaly_data['anomalies']:
                anomalies_text = self._extract_anomalies_text(anomaly_data['anomalies'])
                
                # Get severity assessment
                severity = await self.severity_chain.arun(anomalies=anomalies_text)
                analysis_dict['detailed_severity'] = severity
                
                # Get root causes
                root_causes = await self.root_cause_chain.arun(anomalies=anomalies_text)
                analysis_dict['detailed_root_causes'] = root_causes
            
            return {
                'success': True,
                'analysis': analysis_dict,
                'raw_anomaly_data': anomaly_data
            }
            
        except Exception as e:
            logger.error(f"Anomaly analysis error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_anomaly_data(self, data: Dict[str, Any]) -> str:
        """Format anomaly data for LLM consumption"""
        formatted = []
        
        # Basic info
        if 'total_anomalies' in data:
            formatted.append(f"Total Anomalies: {data['total_anomalies']}")
        
        if 'critical_anomalies' in data:
            formatted.append(f"Critical Anomalies: {data['critical_anomalies']}")
        
        if 'affected_columns' in data:
            formatted.append(f"Affected Columns: {data['affected_columns']}")
        
        # Anomalies by type
        if 'anomaly_types' in data and data['anomaly_types']:
            formatted.append("\n## Anomalies by Type")
            for anomaly_type, count in data['anomaly_types'].items():
                formatted.append(f"- {anomaly_type}: {count}")
        
        # Detailed anomalies
        if 'anomalies' in data and data['anomalies']:
            formatted.append("\n## Detailed Anomalies")
            
            # Group by anomaly category
            anomalies_by_category = {}
            for category, anomalies in data['anomalies'].items():
                if anomalies:
                    anomalies_by_category[category] = anomalies
            
            for category, anomalies in anomalies_by_category.items():
                formatted.append(f"\n### {category.title()} Anomalies")
                for anomaly in anomalies[:5]:  # Limit to 5 per category
                    col = anomaly.get('column', 'Unknown')
                    desc = anomaly.get('description', 'No description')[:100]
                    severity = anomaly.get('severity', 'unknown').upper()
                    formatted.append(f"- [{severity}] {col}: {desc}")
                
                if len(anomalies) > 5:
                    formatted.append(f"- ... and {len(anomalies) - 5} more")
        
        return "\n".join(formatted)
    
    def _extract_anomalies_text(self, anomalies_dict: Dict[str, List]) -> str:
        """Extract anomalies as text for analysis"""
        text_parts = []
        
        for category, anomalies in anomalies_dict.items():
            if anomalies:
                text_parts.append(f"{category.title()} Anomalies:")
                for anomaly in anomalies[:10]:  # Limit to 10 per category
                    col = anomaly.get('column', 'Unknown')
                    desc = anomaly.get('description', 'No description')
                    severity = anomaly.get('severity', 'unknown')
                    count = anomaly.get('count', 1)
                    
                    text_parts.append(f"  - {col}: {desc} (Severity: {severity}, Count: {count})")
        
        return "\n".join(text_parts)
    
    async def generate_alert_summary(self, anomalies: Dict[str, Any], threshold: float = 0.05) -> str:
        """Generate alert summary for significant anomalies"""
        try:
            # Calculate if anomalies exceed threshold
            total_rows = anomalies.get('total_rows', 1)
            total_anomalies = anomalies.get('total_anomalies', 0)
            anomaly_rate = total_anomalies / total_rows if total_rows > 0 else 0
            
            if anomaly_rate < threshold:
                return "No significant anomalies detected above threshold."
            
            # Generate alert
            prompt = PromptTemplate(
                template="Generate an alert summary for significant anomalies detected:\n\n" +
                "Anomaly Rate: {anomaly_rate:.1%} (threshold: {threshold:.1%})\n" +
                "Total Anomalies: {total_anomalies}\n" +
                "Critical Anomalies: {critical_anomalies}\n\n" +
                "Focus on:\n1. Urgency level\n2. Key affected areas\n3. Immediate actions\n4. Who to notify",
                input_variables=["anomaly_rate", "threshold", "total_anomalies", "critical_anomalies"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            alert = await chain.arun(
                anomaly_rate=anomaly_rate,
                threshold=threshold,
                total_anomalies=total_anomalies,
                critical_anomalies=anomalies.get('critical_anomalies', 0)
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Alert generation error: {str(e)}")
            return "Unable to generate alert summary."
    
    async def suggest_monitoring_rules(self, anomalies: Dict[str, Any]) -> List[str]:
        """Suggest monitoring rules based on detected anomalies"""
        try:
            if not anomalies.get('anomalies'):
                return ["No anomalies detected to base monitoring rules on."]
            
            # Extract patterns from anomalies
            patterns = self._extract_anomaly_patterns(anomalies['anomalies'])
            
            prompt = PromptTemplate(
                template="Based on these anomaly patterns, suggest monitoring rules for detection:\n\n{patterns}\n\n" +
                "Suggest rules in the format:\n- Rule: [description]\n- Threshold: [value]\n- Alert: [type]\n",
                input_variables=["patterns"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            rules_text = await chain.arun(patterns=patterns)
            
            # Parse rules into list
            rules = []
            current_rule = {}
            
            for line in rules_text.split('\n'):
                line = line.strip()
                if line.startswith('- Rule:'):
                    if current_rule:
                        rules.append(current_rule)
                    current_rule = {'rule': line.replace('- Rule:', '').strip()}
                elif line.startswith('- Threshold:'):
                    current_rule['threshold'] = line.replace('- Threshold:', '').strip()
                elif line.startswith('- Alert:'):
                    current_rule['alert'] = line.replace('- Alert:', '').strip()
            
            if current_rule:
                rules.append(current_rule)
            
            # Format as strings
            return [f"{r.get('rule', 'Unknown')} (Threshold: {r.get('threshold', 'N/A')})" 
                   for r in rules[:5]]  # Limit to 5 rules
            
        except Exception as e:
            logger.error(f"Monitoring rules suggestion error: {str(e)}")
            return ["Unable to generate monitoring rules."]
    
    def _extract_anomaly_patterns(self, anomalies_dict: Dict[str, List]) -> str:
        """Extract patterns from anomalies for rule generation"""
        patterns = []
        
        for category, anomalies in anomalies_dict.items():
            if anomalies:
                # Group by column
                columns = {}
                for anomaly in anomalies:
                    col = anomaly.get('column', 'Unknown')
                    if col not in columns:
                        columns[col] = []
                    columns[col].append(anomaly)
                
                # Identify patterns per column
                for col, col_anomalies in columns.items():
                    if len(col_anomalies) > 1:
                        # Multiple anomalies in same column
                        types = set(a.get('anomaly_type', 'unknown') for a in col_anomalies)
                        avg_count = sum(a.get('count', 1) for a in col_anomalies) / len(col_anomalies)
                        
                        patterns.append(
                            f"Column '{col}' had {len(col_anomalies)} anomalies of types: {', '.join(types)}. "
                            f"Average count: {avg_count:.1f}"
                        )
        
        return "\n".join(patterns) if patterns else "No clear patterns detected."