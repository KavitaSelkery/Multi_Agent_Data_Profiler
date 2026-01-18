"""
LangChain chains for SQL generation
"""
from typing import Dict, Any, Optional, Tuple, List
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from loguru import logger

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import Settings
from config.prompts import SQL_AGENT_PROMPT

class GeneratedSQL(BaseModel):
    """Pydantic model for generated SQL output"""
    sql: str = Field(description="The generated SQL query")
    explanation: str = Field(description="Explanation of the query")
    confidence: float = Field(description="Confidence score (0-1)")
    assumptions: List[str] = Field(description="Assumptions made")
    limitations: Optional[str] = Field(description="Any limitations")

class SQLGenerationChain:
    """Chain for SQL generation and optimization"""
    
    def __init__(self, database_context: str = ""):
        """Initialize SQL generation chain"""
        self.settings = Settings()
        self.llm = ChatOpenAI(
            model=self.settings.OPENAI_MODEL,
            temperature=0.1,
            api_key=self.settings.OPENAI_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=GeneratedSQL)
        self.database_context = database_context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self._initialize_chains()
    
    def _initialize_chains(self):
        """Initialize all chains"""
        # Main SQL generation chain
        self.generation_chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                ("system", SQL_AGENT_PROMPT + "\n\n" + self.database_context + "\n\n" +
                 "Return structured output with SQL and explanation."),
                ("human", "Generate SQL for: {query}")
            ]),
            output_key="generated_sql"
        )
        
        # SQL validation chain
        self.validation_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Validate this SQL query for Snowflake compatibility:\n\n```sql\n{sql}\n```\n\n" +
                "Check for:\n1. Syntax errors\n2. Snowflake-specific issues\n3. Performance concerns\n4. Security issues\n" +
                "Return validation results.",
                input_variables=["sql"]
            ),
            output_key="validation"
        )
        
        # SQL optimization chain
        self.optimization_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Optimize this SQL query for Snowflake performance:\n\n```sql\n{sql}\n```\n\n" +
                "Consider:\n1. Query structure\n2. Join optimization\n3. Filter pushdown\n4. Column pruning\n5. Partitioning\n" +
                "Return the optimized SQL and explanation.",
                input_variables=["sql"]
            ),
            output_key="optimized_sql"
        )
        
        # Explanation chain
        self.explanation_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="Explain this SQL query in simple terms:\n\n```sql\n{sql}\n```\n\n" +
                "Focus on:\n1. What the query does\n2. Key operations\n3. Expected output\n4. Performance considerations",
                input_variables=["sql"]
            ),
            output_key="explanation"
        )
    
    async def generate_sql(self, natural_language_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate SQL from natural language query"""
        try:
            # Prepare context
            full_context = self.database_context
            if context:
                if 'table_name' in context:
                    full_context += f"\nTable context: {context['table_name']}"
                if 'column_hints' in context:
                    full_context += f"\nColumn hints: {', '.join(context['column_hints'])}"
            
            # Update prompt with context
            updated_prompt = ChatPromptTemplate.from_messages([
                ("system", SQL_AGENT_PROMPT + "\n\n" + full_context + "\n\n" +
                 "{format_instructions}"),
                ("human", "Generate SQL for: {query}")
            ])
            
            generation_chain = LLMChain(
                llm=self.llm,
                prompt=updated_prompt.partial(
                    format_instructions=self.parser.get_format_instructions()
                ),
                output_key="generated_sql"
            )
            
            # Generate SQL
            result = await generation_chain.arun(query=natural_language_query)
            
            # Parse result
            try:
                parsed_result = self.parser.parse(result)
                sql_output = parsed_result.dict()
            except Exception as parse_error:
                logger.warning(f"Parse error, extracting SQL directly: {parse_error}")
                # Fallback: extract SQL from response
                sql_output = {
                    'sql': self._extract_sql_from_text(result),
                    'explanation': 'Generated from natural language query',
                    'confidence': 0.7,
                    'assumptions': ['Direct SQL extraction'],
                    'limitations': 'Could not parse structured output'
                }
            
            # Validate the SQL
            validation = await self.validation_chain.arun(sql=sql_output['sql'])
            sql_output['validation'] = validation
            
            # Generate explanation if not already present
            if not sql_output.get('explanation') or len(sql_output['explanation']) < 50:
                explanation = await self.explanation_chain.arun(sql=sql_output['sql'])
                sql_output['explanation'] = explanation
            
            return {
                'success': True,
                'sql_output': sql_output,
                'natural_language_query': natural_language_query
            }
            
        except Exception as e:
            logger.error(f"SQL generation error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'natural_language_query': natural_language_query
            }
    
    def _extract_sql_from_text(self, text: str) -> str:
        """Extract SQL from text response"""
        # Look for SQL code blocks
        import re
        
        # Pattern for SQL code blocks
        sql_pattern = r'```sql\n(.*?)\n```'
        matches = re.findall(sql_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Pattern for generic code blocks
        code_pattern = r'```\n(.*?)\n```'
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            # Check if it looks like SQL
            code = matches[0].strip()
            if any(keyword in code.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY']):
                return code
        
        # Last resort: find lines that look like SQL
        lines = text.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('SELECT') or line_stripped.startswith('WITH'):
                in_sql = True
            
            if in_sql and line_stripped and not line_stripped.startswith('--'):
                sql_lines.append(line_stripped)
            
            if in_sql and (line_stripped.endswith(';') or '```' in line):
                break
        
        if sql_lines:
            return ' '.join(sql_lines)
        
        # Return empty if no SQL found
        return "SELECT 'No SQL generated' as error"
    
    async def optimize_sql(self, sql_query: str) -> Dict[str, Any]:
        """Optimize SQL query"""
        try:
            # Get optimization suggestions
            optimization = await self.optimization_chain.arun(sql=sql_query)
            
            # Extract optimized SQL
            optimized_sql = self._extract_sql_from_text(optimization)
            
            # If no optimized SQL found, use original
            if optimized_sql == "SELECT 'No SQL generated' as error":
                optimized_sql = sql_query
                optimization_note = "No optimization suggestions generated."
            else:
                optimization_note = optimization
            
            # Compare with original
            original_lines = len(sql_query.split('\n'))
            optimized_lines = len(optimized_sql.split('\n'))
            
            return {
                'success': True,
                'original_sql': sql_query,
                'optimized_sql': optimized_sql,
                'optimization_notes': optimization_note,
                'improvement': f"Lines reduced from {original_lines} to {optimized_lines}" 
                               if optimized_lines < original_lines else "No line reduction"
            }
            
        except Exception as e:
            logger.error(f"SQL optimization error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_sql': sql_query
            }
    
    async def explain_sql(self, sql_query: str, detail_level: str = "normal") -> str:
        """Explain SQL query at specified detail level"""
        try:
            prompt_template = PromptTemplate(
                template="Explain this SQL query at {detail_level} detail level:\n\n```sql\n{sql}\n```\n\n" +
                "Detail levels:\n- simple: One sentence explanation\n- normal: Paragraph with key points\n" +
                "- detailed: Step-by-step explanation with examples\n- expert: Technical deep dive with optimization tips",
                input_variables=["sql", "detail_level"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            explanation = await chain.arun(sql=sql_query, detail_level=detail_level)
            
            return explanation
            
        except Exception as e:
            logger.error(f"SQL explanation error: {str(e)}")
            return f"Unable to explain SQL: {str(e)}"
    
    async def generate_multiple_options(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple SQL options for the same query"""
        try:
            options = []
            
            for i in range(n):
                # Vary temperature slightly for different options
                temp_llm = ChatOpenAI(
                    model=self.settings.OPENAI_MODEL,
                    temperature=0.1 + (i * 0.1),  # 0.1, 0.2, 0.3
                    api_key=self.settings.OPENAI_API_KEY
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", SQL_AGENT_PROMPT + "\n\n" + self.database_context + "\n\n" +
                     "Generate SQL option " + str(i+1) + " for this query."),
                    ("human", "Generate SQL for: {query}")
                ])
                
                chain = LLMChain(llm=temp_llm, prompt=prompt)
                result = await chain.arun(query=query)
                
                sql = self._extract_sql_from_text(result)
                
                options.append({
                    'option': i + 1,
                    'sql': sql,
                    'temperature': 0.1 + (i * 0.1),
                    'explanation': f"Option {i+1} generated with temperature {0.1 + (i * 0.1):.1f}"
                })
            
            return options
            
        except Exception as e:
            logger.error(f"Multiple options generation error: {str(e)}")
            return [{
                'option': 1,
                'sql': "SELECT 'Error generating options' as error",
                'temperature': 0.1,
                'explanation': f"Error: {str(e)}"
            }]
    
    async def fix_sql_error(self, sql_query: str, error_message: str) -> Dict[str, Any]:
        """Fix SQL error based on error message"""
        try:
            prompt = PromptTemplate(
                template="Fix this SQL query that produced an error:\n\n```sql\n{sql}\n```\n\n" +
                "Error message: {error}\n\n" +
                "Provide:\n1. Fixed SQL\n2. Explanation of the fix\n3. How to avoid similar errors",
                input_variables=["sql", "error"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            fix_result = await chain.arun(sql=sql_query, error=error_message)
            
            # Extract fixed SQL
            fixed_sql = self._extract_sql_from_text(fix_result)
            
            return {
                'success': True,
                'original_sql': sql_query,
                'error_message': error_message,
                'fixed_sql': fixed_sql,
                'fix_explanation': fix_result,
                'suggestions': self._extract_fix_suggestions(fix_result)
            }
            
        except Exception as e:
            logger.error(f"SQL fix error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_sql': sql_query,
                'error_message': error_message
            }
    
    def _extract_fix_suggestions(self, fix_text: str) -> List[str]:
        """Extract fix suggestions from fix explanation"""
        suggestions = []
        lines = fix_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['suggestion', 'recommend', 'tip', 'advice', 'avoid']):
                suggestions.append(line.strip())
        
        return suggestions[:5]  # Return top 5 suggestions