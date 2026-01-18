"""
Orchestrator for coordinating multiple agents and workflows
"""
from typing import Dict, Any, Optional, List
import asyncio
from loguru import logger
import pandas as pd
from datetime import datetime
import sys,os
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.profiler_agent import ProfilerAgent
from agents.anomaly_agent import AnomalyAgent
from agents.sql_agent import SQLAgent
from agents.reporter_agent import ReporterAgent
from tools.snowflake_tools import SnowflakeManager
from config.settings import Settings

class WorkflowOrchestrator:
    """Orchestrates multiple agents and workflows"""
    
    def __init__(self, snowflake_manager: SnowflakeManager = None, openai_api_key: str = None):
        """Initialize orchestrator with agents"""
        self.snowflake_manager = snowflake_manager
        self.settings = Settings()
        
        if openai_api_key:
            self.settings.OPENAI_API_KEY = openai_api_key
        
        # Initialize agents as None initially
        self.agents = {
            'profiler': None,
            'anomaly': None,
            'sql': None,
            'reporter': None
        }
        
        self._initialized = False
        self._initialization_error = None
        self._lock = threading.Lock()  # For thread safety
        
    def initialize(self) -> bool:
        """Initialize all agents synchronously"""
        with self._lock:
            if self._initialized:
                logger.info("Orchestrator already initialized")
                return True
                
            try:
                logger.info("Initializing orchestrator...")
                
                if not self.snowflake_manager:
                    raise ValueError("SnowflakeManager is required for initialization")
                
                if not self.settings.OPENAI_API_KEY:
                    logger.warning("OpenAI API key not set. Agents may not function properly.")
                
                # Initialize agents one by one with error handling
                agents_to_initialize = [
                    ('profiler', ProfilerAgent),
                    ('anomaly', AnomalyAgent),
                    ('sql', SQLAgent),
                    ('reporter', ReporterAgent)
                ]
                
                successful_agents = 0
                for agent_name, agent_class in agents_to_initialize:
                    try:
                        logger.info(f"Initializing {agent_name} agent...")
                        self.agents[agent_name] = agent_class(self.snowflake_manager)
                        successful_agents += 1
                        logger.info(f"✅ {agent_name.title()} Agent initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize {agent_name} agent: {str(e)}")
                        self.agents[agent_name] = None
                
                # Check if any agents were successfully initialized
                if successful_agents == 0:
                    raise Exception("No agents could be initialized")
                
                self._initialized = True
                logger.info(f"✅ Orchestrator initialized with {successful_agents}/4 agents")
                return True
                
            except Exception as e:
                self._initialization_error = str(e)
                logger.error(f"❌ Orchestrator initialization failed: {str(e)}")
                return False
    
    async def initialize_async(self) -> bool:
        """Initialize all agents asynchronously"""
        if self._initialized:
            return True
            
        try:
            logger.info("Initializing orchestrator asynchronously...")
            
            if not self.snowflake_manager:
                raise ValueError("SnowflakeManager is required for initialization")
            
            # Initialize agents sequentially with error handling (simpler than concurrent)
            agents_to_initialize = [
                ('profiler', ProfilerAgent),
                ('anomaly', AnomalyAgent),
                ('sql', SQLAgent),
                ('reporter', ReporterAgent)
            ]
            
            successful_agents = 0
            for agent_name, agent_class in agents_to_initialize:
                try:
                    logger.info(f"Initializing {agent_name} agent...")
                    self.agents[agent_name] = agent_class(self.snowflake_manager)
                    successful_agents += 1
                    logger.info(f"✅ {agent_name.title()} Agent initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize {agent_name} agent: {str(e)}")
                    self.agents[agent_name] = None
            
            # Check if any agents were successfully initialized
            if successful_agents == 0:
                raise Exception("No agents could be initialized")
            
            self._initialized = True
            logger.info(f"✅ Orchestrator initialized with {successful_agents}/4 agents")
            return True
            
        except Exception as e:
            self._initialization_error = str(e)
            logger.error(f"❌ Orchestrator initialization failed: {str(e)}")
            return False
    
    def _ensure_initialized(self):
        """Ensure orchestrator is initialized, raise error if not"""
        if not self._initialized:
            error_msg = "Orchestrator not initialized. Please call initialize() first."
            if self._initialization_error:
                error_msg += f" Previous error: {self._initialization_error}"
            raise RuntimeError(error_msg)
        
        # Check if we have at least one working agent
        working_agents = [name for name, agent in self.agents.items() if agent is not None]
        if not working_agents:
            raise RuntimeError("Orchestrator has no working agents")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {}
        for name, agent in self.agents.items():
            if agent:
                status[name] = {
                    'initialized': True,
                    'tools_count': len(agent.tools) if hasattr(agent, 'tools') and agent.tools else 0,
                    'has_executor': hasattr(agent, 'agent_executor') and agent.agent_executor is not None,
                    'name': getattr(agent, 'name', 'Unknown'),
                    'description': getattr(agent, 'description', 'No description')
                }
            else:
                status[name] = {
                    'initialized': False,
                    'error': 'Failed to initialize'
                }
        
        status['orchestrator'] = {
            'initialized': self._initialized,
            'error': self._initialization_error,
            'working_agents': len([a for a in self.agents.values() if a is not None])
        }
        
        return status
    
    async def run_profiling_workflow(self, table_name: str) -> Dict[str, Any]:
        """Run data profiling workflow"""
        self._ensure_initialized()
        
        if not self.agents['profiler']:
            return {
                'success': False,
                'error': 'Profiler agent not available',
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"Starting profiling workflow for: {table_name}")
        
        try:
            result = await self.agents['profiler'].run({
                'table_name': table_name
            })
            return result
        except Exception as e:
            logger.error(f"Profiling workflow error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_anomaly_detection_workflow(self, table_name: str) -> Dict[str, Any]:
        """Run anomaly detection workflow"""
        self._ensure_initialized()
        
        if not self.agents['anomaly']:
            return {
                'success': False,
                'error': 'Anomaly agent not available',
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"Starting anomaly detection workflow for: {table_name}")
        
        try:
            result = await self.agents['anomaly'].run({
                'table_name': table_name
            })
            return result
        except Exception as e:
            logger.error(f"Anomaly detection workflow error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_sql_generation_workflow(self, query: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Run SQL generation workflow"""
        self._ensure_initialized()
        
        if not self.agents['sql']:
            return {
                'success': False,
                'error': 'SQL agent not available',
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"Starting SQL generation workflow for query: {query[:50]}...")
        
        try:
            input_data = {'query': query}
            if table_name:
                input_data['table_name'] = table_name
            
            result = await self.agents['sql'].run(input_data)
            return result
        except Exception as e:
            logger.error(f"SQL generation workflow error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def run_report_generation_workflow(self, table_name: str) -> Dict[str, Any]:
        """Run report generation workflow"""
        self._ensure_initialized()
        
        if not self.agents['reporter']:
            return {
                'success': False,
                'error': 'Reporter agent not available',
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"Starting report generation workflow for: {table_name}")
        
        try:
            result = await self.agents['reporter'].run({
                'table_name': table_name
            })
            return result
        except Exception as e:
            logger.error(f"Report generation workflow error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_sync(self, coroutine):
        """Run async coroutine synchronously"""
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Check if loop is already running
            if loop.is_running():
                # This is a tricky situation - we're in an async context already
                # Create a task and wait for it using a different approach
                async def run_in_existing_loop():
                    return await coroutine
                
                # Create and run the task
                task = loop.create_task(run_in_existing_loop())
                
                # Try to run until complete
                try:
                    return loop.run_until_complete(task)
                except RuntimeError as e:
                    if "This event loop is already running" in str(e):
                        # We need to run in a separate thread
                        return self._run_in_separate_thread(coroutine)
                    else:
                        raise
            else:
                # Loop is not running, we can use it directly
                return loop.run_until_complete(coroutine)
                
        except Exception as e:
            logger.error(f"Error running coroutine synchronously: {str(e)}")
            # Fallback to thread-based execution
            return self._run_in_separate_thread(coroutine)
    
    def _run_in_separate_thread(self, coroutine):
        """Run coroutine in a separate thread (fallback method)"""
        result = None
        exception = None
        
        async def run_coroutine():
            nonlocal result, exception
            try:
                result = await coroutine
            except Exception as e:
                exception = e
        
        # Create new event loop in a separate thread
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_coroutine())
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()
        
        if exception:
            raise exception
        return result
    
    def initialize_sync(self) -> bool:
        """Initialize synchronously"""
        return self.initialize()  # Use the synchronous initialize method
    
    def is_initialized(self) -> bool:
        """Check if orchestrator is initialized"""
        return self._initialized
    
    def get_available_agents(self) -> List[str]:
        """Get list of available (initialized) agents"""
        return [name for name, agent in self.agents.items() if agent is not None]
    
    def set_snowflake_manager(self, snowflake_manager: SnowflakeManager):
        """Set snowflake manager after initialization"""
        if self._initialized:
            logger.warning("Snowflake manager changed after initialization. Agents may need reinitialization.")
        self.snowflake_manager = snowflake_manager
    
    def set_openai_api_key(self, api_key: str):
        """Set OpenAI API key after initialization"""
        if self._initialized:
            logger.warning("OpenAI API key changed after initialization. Some agents may need reinitialization.")
        self.settings.OPENAI_API_KEY = api_key
    
    def reset(self):
        """Reset orchestrator to uninitialized state"""
        with self._lock:
            self.agents = {
                'profiler': None,
                'anomaly': None,
                'sql': None,
                'reporter': None
            }
            self._initialized = False
            self._initialization_error = None
            logger.info("Orchestrator reset")
    
    def __repr__(self):
        """String representation of orchestrator"""
        status = "initialized" if self._initialized else "not initialized"
        working_agents = len(self.get_available_agents())
        return f"WorkflowOrchestrator(status={status}, working_agents={working_agents}/4)"