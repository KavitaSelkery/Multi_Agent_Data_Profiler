# Data Quality & Analysis Agent Suite

## 📊 Overview

A sophisticated, enterprise-grade data quality analysis platform built with **LangChain** and **LangGraph**. This suite provides intelligent agents for automated data profiling, anomaly detection, SQL generation, and comprehensive reporting with workflow orchestration capabilities.

## ✨ Key Features

### 🤖 **Intelligent Agents**
- **Profiler Agent**: Automated schema discovery, statistical analysis, and data quality assessment
- **Anomaly Agent**: Advanced outlier detection using IQR, Z-Score, and Isolation Forest algorithms
- **SQL Agent**: Natural language to SQL conversion with validation, optimization, and execution
- **Reporter Agent**: Automated report generation with insights and recommendations

### 🔄 **Workflow Orchestration**
- **LangGraph Integration**: Stateful workflow management with conditional routing
- **Async/Sync Execution**: Flexible execution modes for different environments
- **Agent Coordination**: Seamless collaboration between specialized agents
- **Error Handling**: Robust error recovery and graceful degradation

### 💾 **Advanced Infrastructure**
- **Vector Store Management**: ChromaDB integration for semantic memory and retrieval
- **Caching Layer**: Redis-based caching with LangChain compatibility
- **Conversation Memory**: Persistent memory management across sessions
- **Snowflake Integration**: Native Snowflake connectivity with query optimization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   API / CLI Layer                    │
└─────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────┐
│               Workflow Orchestrator                  │
│           (State Management & Routing)               │
└─────────────────────────────────────────────────────┘
                           │
    ┌───────────┬───────────┬───────────┬───────────┐
    │           │           │           │           │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Profiler│ │Anomaly│ │ SQL   │ │Report │ │Custom │
│ Agent │ │ Agent │ │ Agent  │ │ Agent │ │Agent  │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
    │           │           │           │           │
┌─────────────────────────────────────────────────────┐
│              Core Tools & Utilities                  │
│  • Data Analysis   • Snowflake Connectors           │
│  • Vector Stores   • Cache Management               │
│  • Memory Systems  • Validation Chains              │
└─────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd data-quality-agent-suite

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Configuration

Create a `.env` file with:

```env
# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema

# OpenAI Configuration
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4-turbo-preview

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379/0

# Vector Store Path
VECTOR_STORE_PATH=./data/vector_store
```

### Basic Usage

```python
from tools.snowflake_tools import SnowflakeManager
from orchestrator import WorkflowOrchestrator

# Initialize Snowflake connection
snowflake = SnowflakeManager()
snowflake.connect()  # Uses environment variables

# Create orchestrator
orchestrator = WorkflowOrchestrator(snowflake_manager=snowflake)
orchestrator.initialize()

# Run data profiling
result = orchestrator.run_sync(
    orchestrator.run_profiling_workflow("your_table_name")
)
```

## 🛠️ Core Components

### 1. **Agents**

#### Profiler Agent
- **Purpose**: Automated data profiling and quality assessment
- **Tools**:
  - Schema discovery and validation
  - Statistical analysis (min, max, mean, distribution)
  - Data quality scoring
  - Constraint validation

#### Anomaly Agent
- **Purpose**: Detect outliers and unusual patterns
- **Methods**:
  - IQR (Interquartile Range)
  - Z-Score analysis
  - Isolation Forest
  - Temporal pattern detection

#### SQL Agent
- **Purpose**: Natural language to SQL conversion
- **Capabilities**:
  - Query generation from natural language
  - SQL validation and optimization
  - Query explanation
  - Error debugging

### 2. **Workflow System**

#### LangGraph Integration
```python
from workflow_graph import DataQualityWorkflow

# Create workflow graph
workflow = DataQualityWorkflow(agents=orchestrator.agents)

# Execute workflow
result = await workflow.run({
    'table_name': 'sales_data',
    'natural_language_query': 'Find top 10 customers by revenue',
    'workflow_type': 'sql'  # profiling, anomaly, sql, report
})
```

#### State Management
```python
from state import WorkflowStateManager

# Track workflow execution
state_manager = WorkflowStateManager()
execution = state_manager.create_execution(
    workflow_id='profiling_001',
    input_data={'table_name': 'customers'}
)
```

### 3. **Data Tools**

#### Data Analyzer
```python
from tools.data_tools import DataAnalyzer

analyzer = DataAnalyzer()
issues = analyzer.analyze_data_quality(df, schema_df)
anomalies = analyzer.detect_anomalies(df, method='IQR')
```

#### Vector Store
```python
from tools.vector_store import VectorStoreManager

vector_store = VectorStoreManager(collection_name='business_data')
vector_store.add_documents(documents, metadatas)
results = vector_store.search(query="customer segmentation", n_results=5)
```

### 4. **Caching & Memory**

#### Redis Cache
```python
from cache import CacheManager

cache = CacheManager(namespace="query_cache")
cached_result = cache.get(query_hash)
if not cached_result:
    result = execute_expensive_operation()
    cache.set(query_hash, result, ttl_seconds=3600)
```

#### Conversation Memory
```python
from conversation_memory import ConversationMemory

memory = ConversationMemory(session_id="user_123")
memory.add_message("human", "Show me sales trends")
memory.add_message("ai", "Sales increased by 15% last quarter")
context = memory.get_memory_context(recent_n=10)
```

## 📋 Workflow Examples

### 1. **Comprehensive Data Quality Check**

```python
async def comprehensive_data_quality(table_name):
    """Run full data quality assessment"""
    
    # 1. Profile the data
    profiling = await orchestrator.run_profiling_workflow(table_name)
    
    # 2. Detect anomalies
    anomalies = await orchestrator.run_anomaly_detection_workflow(table_name)
    
    # 3. Generate report
    report = await orchestrator.run_report_generation_workflow(table_name)
    
    # 4. Create recommendations
    recommendations = generate_recommendations(profiling, anomalies)
    
    return {
        'profiling': profiling,
        'anomalies': anomalies,
        'report': report,
        'recommendations': recommendations
    }
```

### 2. **Natural Language Query Processing**

```python
async def process_natural_language_query(query, context_table=None):
    """Process natural language query with context"""
    
    # Generate SQL from natural language
    sql_result = await orchestrator.run_sql_generation_workflow(
        query=query,
        table_name=context_table
    )
    
    if sql_result['success']:
        # Execute the generated SQL
        sql_query = sql_result['generated_sql']
        df = snowflake_manager.execute_query(sql_query)
        
        # Generate explanation
        explanation = await sql_agent.tools[3].func(sql_query)  # explain_sql tool
        
        return {
            'query': query,
            'generated_sql': sql_query,
            'results': df.to_dict('records'),
            'explanation': explanation,
            'row_count': len(df)
        }
```

### 3. **Automated Monitoring Workflow**

```python
async def automated_data_monitoring(tables, schedule="daily"):
    """Automated data quality monitoring"""
    
    results = {}
    
    for table in tables:
        # Run profiling
        profile = await orchestrator.run_profiling_workflow(table)
        
        # Check for critical issues
        critical_issues = [
            issue for issue in profile.get('data_quality_issues', [])
            if issue.get('severity') == 'critical'
        ]
        
        # Detect anomalies
        anomalies = await orchestrator.run_anomaly_detection_workflow(table)
        
        # Store results
        results[table] = {
            'profile_summary': {
                'row_count': profile.get('row_count'),
                'quality_score': profile.get('data_quality_score'),
                'critical_issues': len(critical_issues)
            },
            'anomalies_detected': anomalies.get('total_anomalies', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Alert if critical issues found
        if critical_issues:
            send_alert(table, critical_issues)
    
    return results
```

## 🔧 Advanced Configuration

### Custom Agent Creation

```python
from agents.base_agent import BaseAgent
from langchain_core.tools import Tool

class CustomDataAgent(BaseAgent):
    """Custom agent for specialized data tasks"""
    
    def __init__(self, snowflake_manager):
        tools = [
            self._create_custom_tool(),
            self._create_analysis_tool()
        ]
        
        super().__init__(
            name="Custom Data Agent",
            description="Specialized agent for custom data operations",
            tools=tools
        )
    
    def _create_custom_tool(self):
        def custom_analysis(table_name, parameters):
            """Custom analysis function"""
            # Your implementation here
            pass
        
        return Tool(
            name="custom_analysis",
            func=custom_analysis,
            description="Perform custom data analysis"
        )
```

### Workflow Customization

```python
from workflow_graph import DataQualityWorkflow
from langgraph.graph import StateGraph

class CustomWorkflow(DataQualityWorkflow):
    """Extended workflow with custom nodes"""
    
    def _create_workflow(self):
        workflow = super()._create_workflow()
        
        # Add custom node
        workflow.add_node("custom_analysis", self.custom_analysis)
        
        # Modify routing
        workflow.add_edge("validate_input", "custom_analysis")
        workflow.add_edge("custom_analysis", "route_workflow")
        
        return workflow
    
    async def custom_analysis(self, state):
        """Custom analysis logic"""
        # Your custom logic here
        state['custom_results'] = perform_custom_analysis(state['table_name'])
        return state
```

## 📈 Performance Optimization

### Caching Strategies

```python
from cache import cache_function

@cache_function(ttl_seconds=300)  # Cache for 5 minutes
async def expensive_data_operation(table_name, parameters):
    """Expensive operation with caching"""
    return await process_large_dataset(table_name, parameters)

# LLM response caching
cache_manager.cache_llm_generation(
    prompt="What is data quality?",
    llm_string="gpt-4-turbo",
    generation="Data quality refers to..."
)
```

### Vector Store Optimization

```python
# Batch document insertion
vector_store.add_documents(
    documents=large_document_set,
    metadatas=[{'source': 'report', 'date': '2024-01-15'} for _ in large_document_set],
    batch_size=100  # Process in batches
)

# Semantic search with filters
results = vector_store.search(
    query="quarterly financial performance",
    n_results=10,
    filter_metadata={'year': 2024, 'department': 'finance'}
)
```

## 🧪 Testing

### Unit Tests

```python
# test_agents.py
import pytest
from agents.profiler_agent import ProfilerAgent

@pytest.mark.asyncio
async def test_profiler_agent():
    """Test profiler agent functionality"""
    agent = ProfilerAgent(mock_snowflake_manager)
    result = await agent.run({'table_name': 'test_table'})
    
    assert result['success'] == True
    assert 'table_name' in result
    assert 'analysis' in result
```

### Integration Tests

```python
# test_workflows.py
@pytest.mark.integration
async def test_complete_workflow():
    """Test complete workflow execution"""
    orchestrator = WorkflowOrchestrator(test_snowflake_manager)
    await orchestrator.initialize_async()
    
    results = await asyncio.gather(
        orchestrator.run_profiling_workflow('sales'),
        orchestrator.run_anomaly_detection_workflow('sales'),
        orchestrator.run_sql_generation_workflow('top 10 products')
    )
    
    assert all(r['success'] for r in results)
```

## 📚 API Reference

### Orchestrator Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `initialize()` | Initialize all agents | `bool` |
| `run_profiling_workflow(table)` | Profile table data | `Dict` |
| `run_anomaly_detection_workflow(table)` | Detect anomalies | `Dict` |
| `run_sql_generation_workflow(query)` | Generate SQL | `Dict` |
| `run_report_generation_workflow(table)` | Generate report | `Dict` |
| `get_agent_status()` | Agent status report | `Dict` |

### Agent Tools

Each agent provides specialized tools:

- **Profiler Agent**: `get_table_schema`, `get_table_stats`, `analyze_data_quality`, `check_constraints`
- **Anomaly Agent**: `detect_numeric_anomalies`, `detect_temporal_anomalies`, `detect_categorical_anomalies`, `analyze_anomaly_causes`
- **SQL Agent**: `generate_sql`, `validate_sql`, `execute_sql`, `explain_sql`, `optimize_sql`

## 🔒 Security Considerations

### Environment Security
- Store secrets in environment variables or secure vaults
- Use separate credentials for development/production
- Implement API key rotation policies

### Data Protection
- Implement data masking for sensitive information
- Use read-only database connections where possible
- Audit all data access and modifications

### Network Security
- Encrypt data in transit (TLS/SSL)
- Restrict network access to required services only
- Implement rate limiting and DDoS protection

## 📊 Monitoring & Logging

### Structured Logging
```python
from loguru import logger
import json

# Structured logging
logger.info("Workflow completed", extra={
    'workflow_id': workflow_id,
    'duration_seconds': duration,
    'table_name': table_name,
    'row_count': row_count
})

# Export logs for analysis
logger.add("logs/data_quality_{time}.json", serialize=True)
```

### Performance Metrics
```python
# Track execution metrics
metrics = {
    'agent_execution_times': {},
    'cache_hit_rates': cache_manager.get_stats()['hit_rate'],
    'query_response_times': [],
    'error_rates': calculate_error_rate()
}

# Export to monitoring system
export_metrics_to_prometheus(metrics)
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-quality-agent
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: agent
        image: data-quality-agent:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .

# Type checking
mypy .
```

## 📞 Support
- **Email**: kselkari@gmail.com

---

## 🎯 Roadmap

### Short Term (Q1 2026)
- [ ] Additional database connectors (PostgreSQL, BigQuery)
- [ ] Enhanced visualization capabilities
- [ ] More anomaly detection algorithms
- [ ] Improved caching strategies
- [ ] Addition Humand Feedback in Loop (RLHF) for agents
- [ ] Enhanced security features
- [ ] Improved Concurrency Handling

### Medium Term (Q2 2026)
- [ ] Real-time data quality monitoring
- [ ] Automated remediation suggestions
- [ ] Integration with data catalogs
- [ ] Advanced ML-based anomaly detection

### Long Term (H2 2026)
- [ ] Predictive data quality forecasting
- [ ] Automated data governance
- [ ] Multi-tenant support
- [ ] Advanced workflow customization

---


**Built with ❤️ by Kavita Selkery -- Solution Architect
