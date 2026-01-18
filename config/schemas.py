"""
Extended schemas for Graph workflows and state management
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import json

# Import existing schemas if they exist
try:
    from schemas import (
        DataQualityIssueSeverity, 
        AgentRole, 
        DataQualityIssue,
        Agent
    )
except ImportError:
    # Define minimal versions if original schemas don't exist
    class DataQualityIssueSeverity(str, Enum):
        CRITICAL = "critical"
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"
        INFO = "info"
    
    class AgentRole(str, Enum):
        DATA_STEWARD = "data_steward"
        DATA_ENGINEER = "data_engineer"
        DATA_ANALYST = "data_analyst"
        DATA_SCIENTIST = "data_scientist"
    
    class DataQualityIssue(BaseModel):
        issue_id: str
        title: str
        severity: DataQualityIssueSeverity
    
    class Agent(BaseModel):
        agent_id: str
        username: str
        role: AgentRole

class WorkflowType(str, Enum):
    """Types of workflows"""
    PROFILING = "profiling"
    ANOMALY_DETECTION = "anomaly"
    SQL_GENERATION = "sql"
    REPORT_GENERATION = "report"
    COMPREHENSIVE = "comprehensive"
    CUSTOM = "custom"

class WorkflowStatus(str, Enum):
    """Workflow status"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DISABLED = "disabled"

class NodeType(str, Enum):
    """Types of workflow nodes"""
    START = "start"
    END = "end"
    TASK = "task"
    DECISION = "decision"
    PARALLEL = "parallel"
    AGENT = "agent"
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"

class WorkflowNodeSchema(BaseModel):
    """Schema for workflow node"""
    node_id: str = Field(..., description="Unique node identifier")
    node_type: NodeType = Field(..., description="Type of node")
    label: str = Field(..., description="Human-readable label")
    description: Optional[str] = Field(None, description="Node description")
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Node configuration")
    agent_id: Optional[str] = Field(None, description="Assigned agent ID")
    tool_name: Optional[str] = Field(None, description="Tool to execute")
    
    # Position in graph
    position_x: float = Field(0.0, description="X position in workflow editor")
    position_y: float = Field(0.0, description="Y position in workflow editor")
    
    # Connections
    incoming_edges: List[str] = Field(default_factory=list, description="Incoming edge IDs")
    outgoing_edges: List[str] = Field(default_factory=list, description="Outgoing edge IDs")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None)
    created_by: str = Field("system", description="Creator ID")
    
    @validator('config')
    def validate_config(cls, v):
        """Validate config is JSON serializable"""
        try:
            json.dumps(v)
        except TypeError:
            raise ValueError("Config must be JSON serializable")
        return v

class WorkflowEdgeSchema(BaseModel):
    """Schema for workflow edge"""
    edge_id: str = Field(..., description="Unique edge identifier")
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")
    label: Optional[str] = Field(None, description="Edge label")
    condition: Optional[str] = Field(None, description="Condition for traversal")
    priority: int = Field(1, ge=1, le=10, description="Edge priority")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowSchema(BaseModel):
    """Schema for workflow definition"""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    workflow_type: WorkflowType = Field(..., description="Type of workflow")
    status: WorkflowStatus = Field(WorkflowStatus.DRAFT, description="Workflow status")
    version: str = Field("1.0.0", description="Workflow version")
    
    # Graph structure
    nodes: Dict[str, WorkflowNodeSchema] = Field(default_factory=dict, description="Workflow nodes")
    edges: Dict[str, WorkflowEdgeSchema] = Field(default_factory=dict, description="Workflow edges")
    start_node_id: Optional[str] = Field(None, description="Start node ID")
    end_node_ids: List[str] = Field(default_factory=list, description="End node IDs")
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Workflow configuration")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    timeout_seconds: Optional[int] = Field(None, ge=1, description="Timeout in seconds")
    concurrent_executions: int = Field(1, ge=1, description="Maximum concurrent executions")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None)
    created_by: str = Field("system", description="Creator ID")
    owner_team: Optional[str] = Field(None, description="Owning team")
    
    class Config:
        schema_extra = {
            "example": {
                "workflow_id": "dq_profiling_001",
                "name": "Data Quality Profiling",
                "description": "Comprehensive data profiling workflow",
                "workflow_type": "profiling",
                "version": "1.0.0",
                "config": {
                    "sample_size": 10000,
                    "enable_anomaly_detection": True
                }
            }
        }

class ExecutionInputSchema(BaseModel):
    """Schema for workflow execution input"""
    workflow_id: str = Field(..., description="Workflow to execute")
    input_data: Dict[str, Any] = Field(..., description="Input data for workflow")
    priority: int = Field(1, ge=1, le=5, description="Execution priority")
    callback_url: Optional[str] = Field(None, description="Callback URL for completion")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
    
    @validator('input_data')
    def validate_input_data(cls, v, values):
        """Validate input data based on workflow type"""
        workflow_id = values.get('workflow_id', '')
        
        # Basic validation for different workflow types
        if 'profiling' in workflow_id.lower() and 'table_name' not in v:
            raise ValueError("Profiling workflows require 'table_name' in input_data")
        
        if 'sql' in workflow_id.lower() and 'query' not in v:
            raise ValueError("SQL generation workflows require 'query' in input_data")
        
        return v

class ExecutionOutputSchema(BaseModel):
    """Schema for workflow execution output"""
    execution_id: str = Field(..., description="Execution identifier")
    workflow_id: str = Field(..., description="Executed workflow ID")
    status: str = Field(..., description="Execution status")
    start_time: datetime = Field(..., description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution end time")
    duration_seconds: Optional[float] = Field(None, description="Execution duration")
    
    # Results
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    node_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-node results")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings generated")
    
    # Performance metrics
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    resource_usage: Dict[str, Any] = Field(default_factory=dict, description="Resource usage")
    
    # Quality metrics
    data_quality_score: Optional[float] = Field(None, ge=0, le=100, description="Overall data quality score")
    issues_found: List[DataQualityIssue] = Field(default_factory=list, description="Data quality issues found")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    
    class Config:
        schema_extra = {
            "example": {
                "execution_id": "exec_123456",
                "workflow_id": "dq_profiling_001",
                "status": "completed",
                "duration_seconds": 45.2,
                "data_quality_score": 85.5,
                "issues_found": [
                    {"issue_id": "issue_001", "title": "Null values in email column", "severity": "high"}
                ]
            }
        }

class WorkflowTemplateSchema(BaseModel):
    """Schema for workflow templates"""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    workflow_type: WorkflowType = Field(..., description="Workflow type")
    category: str = Field("general", description="Template category")
    
    # Template configuration
    default_config: Dict[str, Any] = Field(default_factory=dict, description="Default configuration")
    required_inputs: List[str] = Field(default_factory=list, description="Required input fields")
    optional_inputs: List[str] = Field(default_factory=list, description="Optional input fields")
    
    # Template content
    workflow_definition: WorkflowSchema = Field(..., description="Workflow definition")
    example_input: Dict[str, Any] = Field(default_factory=dict, description="Example input")
    example_output: Dict[str, Any] = Field(default_factory=dict, description="Example output")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Template tags")
    popularity_score: float = Field(0.0, ge=0, description="Template popularity")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None)
    created_by: str = Field("system", description="Creator ID")

class AgentAssignmentSchema(BaseModel):
    """Schema for agent assignments in workflows"""
    assignment_id: str = Field(..., description="Assignment identifier")
    workflow_id: str = Field(..., description="Workflow ID")
    node_id: str = Field(..., description="Node ID")
    agent_id: str = Field(..., description="Agent ID")
    agent_role: AgentRole = Field(..., description="Agent role")
    
    # Assignment rules
    assignment_rule: str = Field("manual", description="Assignment rule type")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Assignment conditions")
    priority: int = Field(1, ge=1, le=5, description="Assignment priority")
    
    # Status
    is_active: bool = Field(True, description="Whether assignment is active")
    last_assigned: Optional[datetime] = Field(None, description="Last assignment time")
    success_count: int = Field(0, ge=0, description="Successful assignments")
    failure_count: int = Field(0, ge=0, description="Failed assignments")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(None)
    created_by: str = Field("system", description="Creator ID")

class WorkflowPerformanceSchema(BaseModel):
    """Schema for workflow performance tracking"""
    performance_id: str = Field(..., description="Performance record ID")
    workflow_id: str = Field(..., description="Workflow ID")
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")
    
    # Execution metrics
    total_executions: int = Field(0, ge=0, description="Total executions")
    successful_executions: int = Field(0, ge=0, description="Successful executions")
    failed_executions: int = Field(0, ge=0, description="Failed executions")
    avg_duration_seconds: Optional[float] = Field(None, ge=0, description="Average duration")
    min_duration_seconds: Optional[float] = Field(None, ge=0, description="Minimum duration")
    max_duration_seconds: Optional[float] = Field(None, ge=0, description="Maximum duration")
    
    # Quality metrics
    avg_data_quality_score: Optional[float] = Field(None, ge=0, le=100, description="Average quality score")
    avg_issues_found: Optional[float] = Field(None, ge=0, description="Average issues found")
    critical_issue_rate: Optional[float] = Field(None, ge=0, le=1, description="Critical issue rate")
    
    # Resource metrics
    avg_memory_mb: Optional[float] = Field(None, ge=0, description="Average memory usage")
    avg_cpu_percent: Optional[float] = Field(None, ge=0, le=100, description="Average CPU usage")
    
    # Trends
    trend: Optional[str] = Field(None, description="Performance trend")
    recommendations: List[str] = Field(default_factory=list, description="Performance recommendations")

# Response models for API
class WorkflowListResponse(BaseModel):
    """Response for listing workflows"""
    workflows: List[WorkflowSchema]
    total_count: int
    page: int
    page_size: int
    has_more: bool

class ExecutionListResponse(BaseModel):
    """Response for listing executions"""
    executions: List[ExecutionOutputSchema]
    total_count: int
    page: int
    page_size: int
    has_more: bool

class WorkflowRunResponse(BaseModel):
    """Response for running a workflow"""
    execution_id: str
    status: str
    estimated_completion_time: Optional[datetime]
    tracking_url: Optional[str]
    message: str

class TemplateListResponse(BaseModel):
    """Response for listing templates"""
    templates: List[WorkflowTemplateSchema]
    total_count: int
    page: int
    page_size: int
    has_more: bool

# Request models
class WorkflowRunRequest(BaseModel):
    """Request for running a workflow"""
    workflow_id: str
    input_data: Dict[str, Any]
    priority: int = 1
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowCreateRequest(BaseModel):
    """Request for creating a workflow"""
    name: str
    description: Optional[str] = None
    workflow_type: WorkflowType
    nodes: Dict[str, WorkflowNodeSchema]
    edges: Dict[str, WorkflowEdgeSchema]
    config: Dict[str, Any] = Field(default_factory=dict)

class WorkflowUpdateRequest(BaseModel):
    """Request for updating a workflow"""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[WorkflowStatus] = None
    config: Optional[Dict[str, Any]] = None
    nodes: Optional[Dict[str, WorkflowNodeSchema]] = None
    edges: Optional[Dict[str, WorkflowEdgeSchema]] = None