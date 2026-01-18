"""
Main Streamlit Application for Agentic Snowflake Data Quality Dashboard
"""
import streamlit as st
from loguru import logger
import sys
import os
import pandas as pd
import asyncio

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from config.settings import Settings
from agents.orchestrator import WorkflowOrchestrator
from ui.components import (
    render_header, 
    render_sidebar, 
    render_profiling_tab,
    render_anomaly_tab,
    render_sql_agent_tab,
    render_dashboard
)

# Setup logging
setup_logger()
logger.info("Starting Snowflake Agentic Dashboard")

# Initialize settings
settings = Settings()

# Page configuration
st.set_page_config(
    page_title="Agentic Snowflake Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .agent-response {
        background-color: #f8f9fa;
        border-left: 4px solid #3B82F6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .agent-thinking {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

class AgenticDashboard:
    """Main dashboard class with agentic capabilities"""
    
    def __init__(self):
        """Initialize the dashboard with agents and state"""
        self.settings = Settings()
        self.orchestrator = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        defaults = {
            'snowflake_connected': False,
            'selected_table': None,
            'profiling_results': {},
            'anomaly_results': {},
            'query_history': [],
            'agent_conversation': [],
            'workflow_state': {},
            'available_tables': [],
            'current_workflow': None,
            'agent_thinking': False,
            'orchestrator_initialized': False  # NEW: Track orchestrator state
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def connect_to_snowflake(self, connection_params):
        """Connect to Snowflake using agentic approach"""
        try:
            from tools.snowflake_tools import SnowflakeManager
            
            # Create SnowflakeManager instance
            snowflake_manager = SnowflakeManager()
            
            # Check what parameters the connect method actually accepts
            # Try with different parameter combinations
            try:
                # Try standard Snowflake connection
                success, message = snowflake_manager.connect(
                    account=connection_params.get('account'),
                    user=connection_params.get('user'),  # Use 'user' not 'username'
                    password=connection_params.get('password'),
                    warehouse=connection_params.get('warehouse'),
                    database=connection_params.get('database'),
                    schema=connection_params.get('schema'),
                    role=connection_params.get('role')
                )
            except TypeError as e:
                # If that fails, try minimal connection
                logger.warning(f"Standard connection failed: {e}. Trying minimal connection...")
                success, message = snowflake_manager.connect(
                    account=connection_params.get('account'),
                    user=connection_params.get('user'),
                    password=connection_params.get('password')
                )
            
            if success:
                st.session_state.snowflake_connected = True
                
                # Try to get tables
                try:
                    tables = snowflake_manager.get_tables()
                    st.session_state.available_tables = tables
                except Exception as e:
                    logger.warning(f"Could not get tables: {str(e)}")
                    st.session_state.available_tables = []
                
                # Initialize orchestrator
                self.orchestrator = WorkflowOrchestrator(
                    snowflake_manager=snowflake_manager,
                    openai_api_key=self.settings.OPENAI_API_KEY
                )
                
                # Try to initialize orchestrator
                try:
                    if hasattr(self.orchestrator, 'initialize'):
                        initialized = self.orchestrator.initialize()
                        st.session_state.orchestrator_initialized = initialized
                        
                        if initialized:
                            # Check agent status
                            status = self.orchestrator.get_agent_status()
                            logger.info(f"Orchestrator initialized with status: {status}")
                            
                            # Add to conversation
                            st.session_state.agent_conversation.append({
                                'role': 'agent',
                                'content': f"✅ Orchestrator initialized successfully! Agents ready.",
                                'type': 'success'
                            })
                        else:
                            st.session_state.agent_conversation.append({
                                'role': 'agent',
                                'content': f"⚠️ Orchestrator initialized but some agents may not be available.",
                                'type': 'warning'
                            })
                    else:
                        # Assume orchestrator auto-initializes
                        st.session_state.orchestrator_initialized = True
                        logger.info("Orchestrator auto-initialized")
                except Exception as e:
                    logger.error(f"Orchestrator initialization failed: {str(e)}")
                    st.session_state.orchestrator_initialized = False
                    st.session_state.agent_conversation.append({
                        'role': 'agent',
                        'content': f"⚠️ Orchestrator setup incomplete: {str(e)}",
                        'type': 'warning'
                    })
                
                return True, "Connected successfully!"
            else:
                return False, message
                
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False, f"Connection failed: {str(e)}"
    
    def run_profiling_workflow(self, table_name):
        """Run data profiling workflow using agents - SYNCHRONOUS version"""
        if not self.orchestrator:
            return None, "Orchestrator not initialized. Please connect first."
        
        if not st.session_state.get('orchestrator_initialized', False):
            return None, "Orchestrator not fully initialized. Please reconnect."
        
        try:
            # Set thinking state
            st.session_state.agent_thinking = True
            
            # Use run_sync if available, otherwise handle async
            if hasattr(self.orchestrator, 'run_sync'):
                # Run profiling workflow synchronously
                result = self.orchestrator.run_sync(
                    self.orchestrator.run_profiling_workflow(table_name)
                )
            else:
                # Fallback: create new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self.orchestrator.run_profiling_workflow(table_name)
                )
                loop.close()
            
            # Store results
            if result:
                st.session_state.profiling_results[table_name] = result
                st.session_state.selected_table = table_name
                
                # Add to conversation history
                st.session_state.agent_conversation.append({
                    'role': 'agent',
                    'content': f"Completed profiling for table: {table_name}",
                    'type': 'success'
                })
            
            return result, None
            
        except Exception as e:
            logger.error(f"Profiling error: {str(e)}")
            error_msg = f"Profiling failed: {str(e)}"
            
            # Add error to conversation
            st.session_state.agent_conversation.append({
                'role': 'agent',
                'content': f"❌ {error_msg}",
                'type': 'error'
            })
            
            return None, error_msg
            
        finally:
            st.session_state.agent_thinking = False
    
    def run_anomaly_detection_workflow(self, table_name):
        """Run anomaly detection workflow using agents - SYNCHRONOUS version"""
        if not self.orchestrator:
            return None, "Orchestrator not initialized. Please connect first."
        
        if not st.session_state.get('orchestrator_initialized', False):
            return None, "Orchestrator not fully initialized. Please reconnect."
        
        try:
            st.session_state.agent_thinking = True
            
            # Use run_sync if available
            if hasattr(self.orchestrator, 'run_sync'):
                result = self.orchestrator.run_sync(
                    self.orchestrator.run_anomaly_detection_workflow(table_name)
                )
            else:
                # Fallback
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self.orchestrator.run_anomaly_detection_workflow(table_name)
                )
                loop.close()
            
            # Store results
            if result:
                st.session_state.anomaly_results[table_name] = result
                
                # Add to conversation
                st.session_state.agent_conversation.append({
                    'role': 'agent',
                    'content': f"Completed anomaly detection for table: {table_name}",
                    'type': 'success'
                })
            
            return result, None
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")
            error_msg = f"Anomaly detection failed: {str(e)}"
            
            st.session_state.agent_conversation.append({
                'role': 'agent',
                'content': f"❌ {error_msg}",
                'type': 'error'
            })
            
            return None, error_msg
            
        finally:
            st.session_state.agent_thinking = False
    
    def run_sql_generation_workflow(self, natural_language_query):
        """Generate and execute SQL using agentic approach - SYNCHRONOUS version"""
        if not self.orchestrator:
            return None, "Orchestrator not initialized. Please connect first."
        
        if not st.session_state.get('orchestrator_initialized', False):
            return None, "Orchestrator not fully initialized. Please reconnect."
        
        try:
            st.session_state.agent_thinking = True
            
            # Use run_sync if available
            if hasattr(self.orchestrator, 'run_sync'):
                result = self.orchestrator.run_sync(
                    self.orchestrator.run_sql_generation_workflow(
                        query=natural_language_query,
                        table_name=st.session_state.selected_table
                    )
                )
            else:
                # Fallback
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self.orchestrator.run_sql_generation_workflow(
                        query=natural_language_query,
                        table_name=st.session_state.selected_table
                    )
                )
                loop.close()
            
            # Add to query history
            if result and result.get('success'):
                st.session_state.query_history.append({
                    'timestamp': result.get('timestamp'),
                    'query': natural_language_query,
                    'sql': result.get('generated_sql'),
                    'result': result.get('result_summary'),
                    'data': result.get('data', pd.DataFrame())
                })
                
                # Add success to conversation
                st.session_state.agent_conversation.append({
                    'role': 'agent',
                    'content': f"✅ SQL generated and executed successfully!",
                    'type': 'success'
                })
            
            return result, None
            
        except Exception as e:
            logger.error(f"SQL generation error: {str(e)}")
            error_msg = f"SQL generation failed: {str(e)}"
            
            st.session_state.agent_conversation.append({
                'role': 'agent',
                'content': f"❌ {error_msg}",
                'type': 'error'
            })
            
            return None, error_msg
            
        finally:
            st.session_state.agent_thinking = False

def main():
    """Main application entry point"""
    dashboard = AgenticDashboard()
    
    # Render header
    render_header("Agentic Snowflake Data Quality Dashboard")
    
    # Sidebar for connection and controls
    with st.sidebar:
        render_sidebar(dashboard)
    
    # Main content area
    if st.session_state.snowflake_connected:
        # Show orchestrator status if available
        if dashboard.orchestrator:
            status = dashboard.orchestrator.get_agent_status()
            
            # Create status indicators
            cols = st.columns(4)
            agent_names = ['profiler', 'anomaly', 'sql', 'reporter']
            
            for idx, agent in enumerate(agent_names):
                with cols[idx]:
                    agent_status = status.get(agent, {})
                    if agent_status.get('initialized', False):
                        st.success(f"🤖 {agent.title()}")
                    else:
                        st.error(f"❌ {agent.title()}")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Agentic Profiling",
            "🔍 Anomaly Detection",
            "🤖 SQL Agent",
            "📈 Dashboard"
        ])
        
        with tab1:
            render_profiling_tab(dashboard)
        
        with tab2:
            render_anomaly_tab(dashboard)
        
        with tab3:
            render_sql_agent_tab(dashboard)
        
        with tab4:
            render_dashboard(dashboard)
    
    else:
        # Show welcome/connection screen
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h2>Welcome to Agentic Snowflake Dashboard</h2>
            <p>Connect to your Snowflake instance to start using AI agents for:</p>
            <ul style='text-align: left; display: inline-block;'>
                <li>🤖 Intelligent data profiling</li>
                <li>🔍 Smart anomaly detection</li>
                <li>💬 Natural language SQL queries</li>
                <li>📊 Automated reporting</li>
                <li>🔄 Agentic workflows</li>
            </ul>
            <p>👉 Enter your credentials in the sidebar to begin</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()