"""
Base agent class with common functionality
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from loguru import logger

# LangChain v1 imports
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import Settings
from config.prompts import AGENT_SYSTEM_PROMPT

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, description: str, tools: List[BaseTool] = None):
        """Initialize base agent"""
        self.name = name
        self.description = description
        self.tools = tools or []
        self.settings = Settings()
        self.llm = None
        self.agent_executor = None
        self.memory = None
        
        self._initialize_llm()
        self._initialize_memory()
        if self.tools:  # Only initialize agent if we have tools
            self._initialize_agent()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        self.llm = ChatOpenAI(
            model=self.settings.OPENAI_MODEL,
            temperature=self.settings.OPENAI_TEMPERATURE,
            max_tokens=self.settings.AGENT_MAX_TOKENS,
            api_key=self.settings.OPENAI_API_KEY
        )
    
    def _initialize_memory(self):
        """Initialize agent memory"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
    
    def _initialize_agent(self):
        """Initialize the agent executor with ReAct logic"""
        try:
            # Create ReAct prompt
            prompt = PromptTemplate.from_template("""
            {system_prompt}
            
            You have access to the following tools:
            
            {tools}
            
            Use the following format:
            
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            Question: {input}
            {agent_scratchpad}
            """)
            
            # Create agent
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt.partial(system_prompt=AGENT_SYSTEM_PROMPT)
            )
            
            # Create executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5,
                early_stopping_method="generate"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {str(e)}")
            self.agent_executor = None
    
    @abstractmethod
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's main logic"""
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate agent input"""
        required_fields = self.get_required_input_fields()
        for field in required_fields:
            if field not in input_data:
                logger.error(f"Missing required field: {field}")
                return False
        return True
    
    @abstractmethod
    def get_required_input_fields(self) -> List[str]:
        """Get list of required input fields"""
        pass
    
    def log_execution(self, input_data: Dict[str, Any], result: Dict[str, Any]):
        """Log agent execution"""
        logger.info(f"Agent {self.name} executed")
        logger.debug(f"Input: {input_data}")
        logger.debug(f"Result: {result}")