"""
State management for LangGraph workflows
"""
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from pydantic import BaseModel, Field
import json

class WorkflowStatus(str, Enum):
    """Workflow status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class NodeStatus(str, Enum):
    """Node status enumeration"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowNode:
    """Represents a node in the workflow"""
    node_id: str
    node_type: str
    status: NodeStatus = NodeStatus.NOT_STARTED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        """Convert to dictionary"""
        result = asdict(self)
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowNode':
        """Create from dictionary"""
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**data)

@dataclass
class WorkflowExecution:
    """Represents a workflow execution"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    nodes: Dict[str, WorkflowNode] = None
    input_data: Dict[str, Any] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.nodes is None:
            self.nodes = {}
    
    def to_dict(self):
        """Convert to dictionary"""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        if self.started_at:
            result['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            result['completed_at'] = self.completed_at.isoformat()
        result['nodes'] = {k: v.to_dict() for k, v in self.nodes.items()}
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowExecution':
        """Create from dictionary"""
        # Handle datetime conversions
        datetime_fields = ['created_at', 'started_at', 'completed_at']
        for field in datetime_fields:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field])
        
        # Convert nodes
        if 'nodes' in data and data['nodes']:
            data['nodes'] = {k: WorkflowNode.from_dict(v) for k, v in data['nodes'].items()}
        
        return cls(**data)

class WorkflowState(TypedDict):
    """
    State definition for LangGraph workflows
    Extends the WorkflowGraph state
    """
    # Basic workflow info
    workflow_id: str
    execution_id: str
    status: WorkflowStatus
    
    # Input data
    table_name: str
    natural_language_query: str
    workflow_type: str  # 'profiling', 'anomaly', 'sql', 'report'
    
    # Processing state
    current_node: str
    previous_node: str
    next_nodes: List[str]
    
    # Node execution tracking
    node_history: List[str]
    node_results: Dict[str, Dict[str, Any]]
    node_errors: Dict[str, str]
    
    # Execution context
    start_time: datetime
    current_iteration: int
    max_iterations: int
    retry_count: int
    
    # Data passed between nodes
    profiling_results: Dict[str, Any]
    anomaly_results: Dict[str, Any]
    sql_results: Dict[str, Any]
    report_results: Dict[str, Any]
    
    # Agent assignments
    assigned_agent: Optional[str]
    agent_assignments: Dict[str, str]  # node_id -> agent_id
    
    # Error handling
    errors: List[str]
    warnings: List[str]
    has_critical_error: bool
    
    # Output
    final_output: Dict[str, Any]
    execution_summary: Dict[str, Any]
    recommendations: List[str]

class WorkflowStateManager:
    """Manages workflow state persistence and retrieval"""
    
    def __init__(self, storage_backend: str = "memory"):
        """Initialize state manager"""
        self.storage_backend = storage_backend
        self.storage = {}  # In-memory storage
        self.executions = {}  # Execution tracking
    
    def create_execution(self, workflow_id: str, input_data: Dict[str, Any]) -> WorkflowExecution:
        """Create a new workflow execution"""
        execution_id = f"{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            input_data=input_data,
            metadata={
                'created_by': 'system',
                'version': '1.0'
            }
        )
        
        self.executions[execution_id] = execution
        return execution
    
    def update_execution(self, execution_id: str, **kwargs):
        """Update execution details"""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            for key, value in kwargs.items():
                if hasattr(execution, key):
                    setattr(execution, key, value)
    
    def update_node_status(self, execution_id: str, node_id: str, 
                          status: NodeStatus, result: Dict[str, Any] = None, 
                          error: str = None):
        """Update node status in execution"""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            
            if node_id not in execution.nodes:
                execution.nodes[node_id] = WorkflowNode(
                    node_id=node_id,
                    node_type=node_id.split('_')[0] if '_' in node_id else node_id
                )
            
            node = execution.nodes[node_id]
            node.status = status
            
            if status == NodeStatus.RUNNING:
                node.start_time = datetime.now()
            elif status in [NodeStatus.SUCCESS, NodeStatus.FAILED, NodeStatus.SKIPPED]:
                node.end_time = datetime.now()
                if node.start_time:
                    node.duration = (node.end_time - node.start_time).total_seconds()
            
            if result:
                node.result = result
            if error:
                node.error = error
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution by ID"""
        return self.executions.get(execution_id)
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowStatus]:
        """Get execution status"""
        execution = self.get_execution(execution_id)
        return execution.status if execution else None
    
    def list_executions(self, workflow_id: str = None, 
                       status: WorkflowStatus = None,
                       limit: int = 100) -> List[WorkflowExecution]:
        """List executions with optional filtering"""
        executions = list(self.executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        # Sort by creation time (newest first)
        executions.sort(key=lambda e: e.created_at, reverse=True)
        
        return executions[:limit]
    
    def save_state(self, execution_id: str, state: WorkflowState):
        """Save workflow state"""
        self.storage[execution_id] = state
    
    def load_state(self, execution_id: str) -> Optional[WorkflowState]:
        """Load workflow state"""
        return self.storage.get(execution_id)
    
    def delete_state(self, execution_id: str):
        """Delete workflow state"""
        if execution_id in self.storage:
            del self.storage[execution_id]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total_executions = len(self.executions)
        successful = len([e for e in self.executions.values() if e.status == WorkflowStatus.COMPLETED])
        failed = len([e for e in self.executions.values() if e.status == WorkflowStatus.FAILED])
        running = len([e for e in self.executions.values() if e.status == WorkflowStatus.RUNNING])
        
        # Calculate average duration
        durations = []
        for e in self.executions.values():
            if e.completed_at and e.started_at:
                durations.append((e.completed_at - e.started_at).total_seconds())
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'total_executions': total_executions,
            'successful': successful,
            'failed': failed,
            'running': running,
            'success_rate': (successful / total_executions * 100) if total_executions > 0 else 0,
            'average_duration_seconds': avg_duration,
            'workflow_types': self._get_workflow_type_distribution()
        }
    
    def _get_workflow_type_distribution(self) -> Dict[str, int]:
        """Get distribution of workflow types"""
        distribution = {}
        for execution in self.executions.values():
            if execution.input_data and 'workflow_type' in execution.input_data:
                wf_type = execution.input_data['workflow_type']
                distribution[wf_type] = distribution.get(wf_type, 0) + 1
        return distribution
    
    def export_execution(self, execution_id: str) -> Dict[str, Any]:
        """Export execution to dictionary"""
        execution = self.get_execution(execution_id)
        if not execution:
            return {}
        
        state = self.load_state(execution_id)
        
        export_data = {
            'execution': execution.to_dict(),
            'state': state if state else {},
            'exported_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        return export_data
    
    def import_execution(self, export_data: Dict[str, Any]) -> str:
        """Import execution from dictionary"""
        try:
            execution = WorkflowExecution.from_dict(export_data['execution'])
            execution_id = execution.execution_id
            
            self.executions[execution_id] = execution
            
            if 'state' in export_data and export_data['state']:
                self.save_state(execution_id, export_data['state'])
            
            return execution_id
            
        except Exception as e:
            raise ValueError(f"Failed to import execution: {str(e)}")