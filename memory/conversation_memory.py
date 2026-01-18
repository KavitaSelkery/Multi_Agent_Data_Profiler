"""
Conversation memory management for agents
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger
from langchain_classic.memory import ConversationBufferMemory#, ChatMessageHistory
#rom langchain_community.chat_message_histories import ChatMessageHistory
# The modern 2026 way for in-memory history:
from langchain_core.chat_history import InMemoryChatMessageHistory as ChatMessageHistory
#from langchain_classic.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings as ChromaSettings
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import Settings
db_path = os.path.abspath(Settings().VECTOR_STORE_PATH)
print(f"Targeting ChromaDB at: {db_path}")



class ConversationMemory:
    """Manages conversation memory for agents"""
    
    def __init__(self, session_id: str = "default"):
        """Initialize conversation memory"""
        self.settings = Settings()
        self.session_id = session_id
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        self.message_history = ChatMessageHistory()
        
        # Initialize vector store for semantic memory
        self.vector_store = None
        self.embeddings = None
        import gc
        del self.vector_store  # Triggers the close of the SQLite connection
        gc.collect()
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize vector store for semantic memory"""
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.settings.EMBEDDING_MODEL,
                api_key=self.settings.OPENAI_API_KEY
            )
            
            # Create persistent vector store
            chroma_settings = ChromaSettings(
                persist_directory=self.settings.VECTOR_STORE_PATH,
                anonymized_telemetry=False
            )
            
            self.vector_store = chromadb.PersistentClient(
                path=self.settings.VECTOR_STORE_PATH,
                settings=chroma_settings
            )
            
            # Get or create collection
            self.collection = self.vector_store.get_or_create_collection(
                name=f"conversation_memory_{self.session_id}",
                metadata={"description": f"Conversation memory for session {self.session_id}"}
            )
            db_path = os.path.abspath(self.settings.VECTOR_STORE_PATH)
            logger.info(f"Vector store initialized for session: {self.session_id}")
            
        except Exception as e:
            logger.warning(f"Vector store initialization failed: {str(e)}")
            print(f"Vector store initialization failed: {str(e)}")
            self.vector_store = None
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to conversation memory"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Create message object
            if role.lower() == 'human':
                message = HumanMessage(content=content)
            elif role.lower() == 'ai':
                message = AIMessage(content=content)
            elif role.lower() == 'system':
                message = SystemMessage(content=content)
            else:
                message = BaseMessage(content=content, type=role)
            
            # Add to memory
            self.memory.chat_memory.add_message(message)
            self.message_history.add_message(message)
            
            # Store in vector store if available
            if self.vector_store and self.embeddings:
                # Generate embedding
                embedding = self.embeddings.embed_query(content)
                
                # Prepare metadata
                msg_metadata = {
                    'role': role,
                    'timestamp': timestamp,
                    'session_id': self.session_id,
                    'content_length': len(content)
                }
                
                if metadata:
                    msg_metadata.update(metadata)
                
                # Add to vector store
                self.collection.add(
                    embeddings=[embedding],
                    metadatas=[msg_metadata],
                    documents=[content],
                    ids=[f"msg_{timestamp}_{len(self.message_history.messages)}"]
                )
            
            logger.debug(f"Added {role} message to memory: {content[:50]}...")
            
        except Exception as e:
            logger.error(f"Error adding message to memory: {str(e)}")
    
    def get_recent_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from memory"""
        try:
            messages = self.message_history.messages[-n:] if self.message_history.messages else []
            
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = 'human'
                elif isinstance(msg, AIMessage):
                    role = 'ai'
                elif isinstance(msg, SystemMessage):
                    role = 'system'
                else:
                    role = 'unknown'
                
                formatted_messages.append({
                    'role': role,
                    'content': msg.content,
                    'type': type(msg).__name__
                })
            
            return formatted_messages
            
        except Exception as e:
            logger.error(f"Error getting recent messages: {str(e)}")
            return []
    
    def search_similar_conversations(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        """Search for similar conversations in memory"""
        try:
            if not self.vector_store or not self.embeddings:
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search vector store
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Format results
            similar_conversations = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    similar_conversations.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'similarity_score': 1 - results['distances'][0][i] if results['distances'] else 0
                    })
            
            return similar_conversations
            
        except Exception as e:
            logger.error(f"Error searching similar conversations: {str(e)}")
            return []
    
    def get_conversation_summary(self) -> str:
        """Generate summary of conversation"""
        try:
            messages = self.get_recent_messages(20)  # Last 20 messages
            
            if not messages:
                return "No conversation history"
            
            # Extract key information
            user_queries = [msg['content'] for msg in messages if msg['role'] == 'human']
            ai_responses = [msg['content'] for msg in messages if msg['role'] == 'ai']
            
            summary = f"""
            Conversation Summary (Session: {self.session_id})
            
            Total Messages: {len(messages)}
            User Queries: {len(user_queries)}
            AI Responses: {len(ai_responses)}
            
            Recent User Queries:
            """
            
            for i, query in enumerate(user_queries[-5:], 1):
                summary += f"\n{i}. {query[:100]}..."
            
            summary += f"\n\nConversation started: {self._get_first_timestamp()}"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _get_first_timestamp(self) -> str:
        """Get timestamp of first message"""
        try:
            if self.vector_store:
                results = self.collection.get(
                    limit=1,
                    include=['metadatas']
                )
                if results['metadatas']:
                    return results['metadatas'][0].get('timestamp', 'Unknown')
        except:
            pass
        
        return "Unknown"
    
    def clear_memory(self):
        """Clear conversation memory"""
        try:
            self.memory.clear()
            self.message_history.clear()
            
            if self.vector_store:
                self.vector_store.reset()
            
            logger.info(f"Cleared memory for session: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
    
    def save_to_file(self, filepath: str):
        """Save conversation memory to file"""
        try:
            messages = self.get_recent_messages(1000)  # Save all messages
            
            memory_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'message_count': len(messages),
                'messages': messages
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved memory to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving memory to file: {str(e)}")
    
    def load_from_file(self, filepath: str):
        """Load conversation memory from file"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Memory file not found: {filepath}")
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Clear existing memory
            self.clear_memory()
            
            # Load messages
            for msg in memory_data.get('messages', []):
                self.add_message(msg['role'], msg['content'])
            
            logger.info(f"Loaded memory from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading memory from file: {str(e)}")
            return False
    
    def get_memory_context(self, recent_n: int = 10, include_summary: bool = True) -> str:
        """Get memory context for LLM"""
        try:
            context_parts = []
            
            if include_summary:
                context_parts.append(self.get_conversation_summary())
            
            # Add recent messages
            recent_messages = self.get_recent_messages(recent_n)
            if recent_messages:
                context_parts.append("\nRecent Conversation:")
                for msg in recent_messages:
                    role_display = "User" if msg['role'] == 'human' else "Assistant"
                    context_parts.append(f"{role_display}: {msg['content'][:200]}...")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting memory context: {str(e)}")
            return "Memory context unavailable."
        
# if __name__ == "__main__":
#     import os
#     print("--- 2026 Diagnostic Start ---")
#     print(f"Current Working Directory: {os.getcwd()}")
    
#     try:
#         # 1. Force check settings first
#         settings = Settings()
#         print(f"Target DB Path: {os.path.abspath(settings.VECTOR_STORE_PATH)}")
        
#         # 2. Initialize Memory
#         print("Initializing Memory...")
#         mem = ConversationMemory(session_id="debug_session_2026")
        
#         if mem.vector_store is None:
#             print("❌ Initialization Failed: Vector store is None. Check API keys/Paths.")
#         else:
#             print("✅ Vector Store Object Created.")
            
#             # 3. Force Write
#             print("Adding message to force disk write...")
#             mem.add_message("human", "Testing persistence for 2026.")
#             print("✅ Message added successfully.")
            
#     except Exception as e:
#         print(f"CRITICAL ERROR: {str(e)}")
    
#     print("--- Diagnostic End ---")

