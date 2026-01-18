"""
LangGraph workflow for data quality analysis
"""
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from loguru import logger
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.profiler_agent import ProfilerAgent
from agents.anomaly_agent import AnomalyAgent
from agents.sql_agent import SQLAgent
from agents.reporter_agent import ReporterAgent

class WorkflowState(TypedDict):
    """State definition for workflow graph"""
    # Input
    table_name: str
    natural_language_query: str
    workflow_type: str  # 'profiling', 'anomaly', 'sql', 'report'
    
    # Processing
    current_step: str
    step_results: Dict[str, Any]
    errors: List[str]
    
    # Output
    profiling_results: Dict[str, Any]
    anomaly_results: Dict[str, Any]
    sql_results: Dict[str, Any]
    report_results: Dict[str, Any]
    final_output: Dict[str, Any]

class DataQualityWorkflow:
    """LangGraph workflow for data quality analysis"""
    
    def __init__(self, agents: Dict[str, Any]):
        """Initialize workflow with agents"""
        self.agents = agents
        self.workflow = self._create_workflow()
        self.checkpointer = MemorySaver()
        self.app = None
    
    def _create_workflow(self) -> StateGraph:
        """Create the workflow graph"""
        # Create graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("validate_input", self.validate_input)
        workflow.add_node("route_workflow", self.route_workflow)
        workflow.add_node("run_profiling", self.run_profiling)
        workflow.add_node("run_anomaly_detection", self.run_anomaly_detection)
        workflow.add_node("run_sql_generation", self.run_sql_generation)
        workflow.add_node("generate_report", self.generate_report)
        workflow.add_node("aggregate_results", self.aggregate_results)
        workflow.add_node("handle_error", self.handle_error)
        
        # Set entry point
        workflow.set_entry_point("validate_input")
        
        # Add edges
        workflow.add_edge("validate_input", "route_workflow")
        
        # Conditional routing
        workflow.add_conditional_edges(
            "route_workflow",
            self.route_condition,
            {
                "profiling": "run_profiling",
                "anomaly": "run_anomaly_detection",
                "sql": "run_sql_generation",
                "report": "generate_report"
            }
        )
        
        # Connect nodes
        workflow.add_edge("run_profiling", "aggregate_results")
        workflow.add_edge("run_anomaly_detection", "aggregate_results")
        workflow.add_edge("run_sql_generation", "aggregate_results")
        workflow.add_edge("generate_report", "aggregate_results")
        workflow.add_edge("aggregate_results", END)
        
        # Error handling
        workflow.add_edge("validate_input", "handle_error", "error")
        workflow.add_edge("route_workflow", "handle_error", "error")
        workflow.add_edge("run_profiling", "handle_error", "error")
        workflow.add_edge("run_anomaly_detection", "handle_error", "error")
        workflow.add_edge("run_sql_generation", "handle_error", "error")
        workflow.add_edge("generate_report", "handle_error", "error")
        workflow.add_edge("handle_error", END)
        
        return workflow
    
    def validate_input(self, state: WorkflowState) -> WorkflowState:
        """Validate input parameters"""
        logger.info("Validating input")
        
        errors = []
        
        # Check workflow type
        valid_workflows = ['profiling', 'anomaly', 'sql', 'report']
        if state['workflow_type'] not in valid_workflows:
            errors.append(f"Invalid workflow type: {state['workflow_type']}")
        
        # Check table name for profiling and anomaly
        if state['workflow_type'] in ['profiling', 'anomaly', 'report']:
            if not state.get('table_name'):
                errors.append("Table name is required for this workflow")
        
        # Check query for SQL generation
        if state['workflow_type'] == 'sql':
            if not state.get('natural_language_query'):
                errors.append("Natural language query is required for SQL generation")
        
        if errors:
            state['errors'] = errors
            return state
        
        state['current_step'] = "input_validated"
        state['step_results'] = {
            'input_validation': {
                'status': 'success',
                'message': 'Input validation passed'
            }
        }
        
        return state
    
    def route_workflow(self, state: WorkflowState) -> WorkflowState:
        """Route to appropriate workflow"""
        logger.info(f"Routing to workflow: {state['workflow_type']}")
        
        state['current_step'] = f"routing_to_{state['workflow_type']}"
        
        return state
    
    def route_condition(self, state: WorkflowState) -> str:
        """Determine which workflow to run"""
        return state['workflow_type']
    
    async def run_profiling(self, state: WorkflowState) -> WorkflowState:
        """Run data profiling workflow"""
        try:
            logger.info(f"Starting profiling for table: {state['table_name']}")
            
            profiler_agent = self.agents.get('profiler')
            if not profiler_agent:
                raise ValueError("Profiler agent not available")
            
            # Run profiling
            result = await profiler_agent.run({
                'table_name': state['table_name']
            })
            
            state['profiling_results'] = result
            state['current_step'] = "profiling_completed"
            state['step_results']['profiling'] = {
                'status': 'success',
                'message': f"Profiling completed for {state['table_name']}"
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Profiling error: {str(e)}")
            state['errors'].append(str(e))
            return state
    
    async def run_anomaly_detection(self, state: WorkflowState) -> WorkflowState:
        """Run anomaly detection workflow"""
        try:
            logger.info(f"Starting anomaly detection for table: {state['table_name']}")
            
            anomaly_agent = self.agents.get('anomaly')
            if not anomaly_agent:
                raise ValueError("Anomaly agent not available")
            
            # Run anomaly detection
            result = await anomaly_agent.run({
                'table_name': state['table_name'],
                'profiling_results': state.get('profiling_results', {})
            })
            
            state['anomaly_results'] = result
            state['current_step'] = "anomaly_detection_completed"
            state['step_results']['anomaly_detection'] = {
                'status': 'success',
                'message': f"Anomaly detection completed for {state['table_name']}"
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")
            state['errors'].append(str(e))
            return state
    
    async def run_sql_generation(self, state: WorkflowState) -> WorkflowState:
        """Run SQL generation workflow"""
        try:
            logger.info(f"Generating SQL for query: {state['natural_language_query']}")
            
            sql_agent = self.agents.get('sql')
            if not sql_agent:
                raise ValueError("SQL agent not available")
            
            # Run SQL generation
            result = await sql_agent.run({
                'query': state['natural_language_query'],
                'table_name': state.get('table_name')
            })
            
            state['sql_results'] = result
            state['current_step'] = "sql_generation_completed"
            state['step_results']['sql_generation'] = {
                'status': 'success',
                'message': f"SQL generated for query"
            }
            
            return state
            
        except Exception as e:
            logger.error(f"SQL generation error: {str(e)}")
            state['errors'].append(str(e))
            return state
    
    async def generate_report(self, state: WorkflowState) -> WorkflowState:
        """Generate report workflow"""
        try:
            logger.info(f"Generating report for table: {state['table_name']}")
            
            reporter_agent = self.agents.get('reporter')
            if not reporter_agent:
                raise ValueError("Reporter agent not available")
            
            # Generate report
            result = await reporter_agent.run({
                'table_name': state['table_name'],
                'profiling_results': state.get('profiling_results', {}),
                'anomaly_results': state.get('anomaly_results', {})
            })
            
            state['report_results'] = result
            state['current_step'] = "report_generated"
            state['step_results']['report_generation'] = {
                'status': 'success',
                'message': f"Report generated for {state['table_name']}"
            }
            
            return state
            
        except Exception as e:
            logger.error(f"Report generation error: {str(e)}")
            state['errors'].append(str(e))
            return state
    
    def aggregate_results(self, state: WorkflowState) -> WorkflowState:
        """Aggregate results from all workflows"""
        logger.info("Aggregating results")
        
        final_output = {
            'workflow_type': state['workflow_type'],
            'table_name': state.get('table_name'),
            'steps_completed': list(state['step_results'].keys()),
            'has_errors': len(state['errors']) > 0,
            'errors': state['errors']
        }
        
        # Add workflow-specific results
        if state['workflow_type'] == 'profiling' and state.get('profiling_results'):
            final_output['profiling'] = state['profiling_results']
        
        if state['workflow_type'] == 'anomaly' and state.get('anomaly_results'):
            final_output['anomaly_detection'] = state['anomaly_results']
        
        if state['workflow_type'] == 'sql' and state.get('sql_results'):
            final_output['sql_generation'] = state['sql_results']
        
        if state['workflow_type'] == 'report' and state.get('report_results'):
            final_output['report'] = state['report_results']
        
        state['final_output'] = final_output
        state['current_step'] = "workflow_completed"
        
        return state
    
    def handle_error(self, state: WorkflowState) -> WorkflowState:
        """Handle errors in workflow"""
        logger.error(f"Workflow error: {state['errors']}")
        
        state['final_output'] = {
            'success': False,
            'errors': state['errors'],
            'workflow_type': state['workflow_type'],
            'current_step': state.get('current_step', 'unknown')
        }
        
        return state
    
    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow"""
        try:
            # Create app if not exists
            if not self.app:
                self.app = self.workflow.compile(checkpointer=self.checkpointer)
            
            # Run workflow
            result = await self.app.ainvoke(
                config,
                config={"configurable": {"thread_id": "user_session_1"}}
            )
            
            return result['final_output']
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }