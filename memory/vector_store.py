"""
Vector store for semantic memory and retrieval
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from loguru import logger
from langchain_openai import OpenAIEmbeddings
#from langchain_core.text_splitter import RecursiveCharacterTextSplitter
# This is the correct way
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import Settings

class VectorStoreManager:
    """Manages vector store for semantic memory"""
    
    def __init__(self, collection_name: str = "default"):
        """Initialize vector store manager"""
        self.settings = Settings()
        self.collection_name = collection_name
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.EMBEDDING_MODEL,
            api_key=self.settings.OPENAI_API_KEY
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize ChromaDB
        self.client = None
        self.collection = None
        self._initialize_chroma()
    
    # def _initialize_chroma(self):
    #     """Initialize ChromaDB client and collection"""
    #     try:
    #         # Create directory if it doesn't exist
    #         os.makedirs(self.settings.VECTOR_STORE_PATH, exist_ok=True)
            
    #         # Configure ChromaDB settings
    #         chroma_settings = ChromaSettings(
    #             persist_directory=self.settings.VECTOR_STORE_PATH,
    #             anonymized_telemetry=False,
    #             is_persistent=True
    #         )
            
    #         # Initialize client
    #         self.client = chromadb.PersistentClient(
    #             path=self.settings.VECTOR_STORE_PATH,
    #             settings=chroma_settings
    #         )
            
    #         # Get or create collection
    #         self.collection = self.client.get_or_create_collection(
    #             name=self.collection_name,
    #             metadata={"description": f"Vector store for {self.collection_name}"}
    #         )
            
    #         logger.info(f"Vector store initialized: {self.collection_name}")
            
    #     except Exception as e:
    #         logger.error(f"Failed to initialize ChromaDB: {str(e)}")
    #         raise
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection"""
        try:
            # 1. Get the absolute path from settings
            db_path = os.path.abspath(self.settings.VECTOR_STORE_PATH)
            
            # 2. Create directory
            os.makedirs(db_path, exist_ok=True)
            
            # 3. Initialize client (Simplified for 2026)
            # Note: Do not pass ChromaSettings unless you have specific 
            # server/telemetry configs; the path is enough for persistence.
            self.client = chromadb.PersistentClient(path=db_path)
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": f"Vector store for {self.collection_name}"}
            )
            
            logger.info(f"✅ ChromaDB persistent at: {db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise


    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add documents to vector store"""
        try:
            if not documents:
                return
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(
                [{"page_content": doc, "metadata": {}} for doc in documents]
            )
            
            # Prepare data for vector store
            chunk_texts = [chunk.page_content for chunk in chunks]
            chunk_ids = [f"chunk_{datetime.now().timestamp()}_{i}" for i in range(len(chunks))]
            
            # Generate embeddings
            chunk_embeddings = self.embeddings.embed_documents(chunk_texts)
            
            # Prepare metadata
            if metadatas and len(metadatas) == len(documents):
                # Distribute metadata to chunks
                chunk_metadatas = []
                for i, chunk in enumerate(chunks):
                    doc_idx = i % len(documents)
                    chunk_metadatas.append({
                        **metadatas[doc_idx],
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'timestamp': datetime.now().isoformat()
                    })
            else:
                chunk_metadatas = [
                    {
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'timestamp': datetime.now().isoformat()
                    }
                    for i in range(len(chunks))
                ]
            
            # Add to collection
            self.collection.add(
                embeddings=chunk_embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
    
    def search(self, query: str, n_results: int = 5, 
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search vector store for similar documents"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search parameters
            search_params = {
                'query_embeddings': [query_embedding],
                'n_results': n_results,
                'include': ['documents', 'metadatas', 'distances']
            }
            
            if filter_metadata:
                search_params['where'] = filter_metadata
            
            # Perform search
            results = self.collection.query(**search_params)
            
            # Format results
            formatted_results = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'similarity_score': 1 - results['distances'][0][i] if results['distances'] else 0,
                        'relevance_score': self._calculate_relevance_score(doc, query)
                    })
            
            # Sort by relevance
            formatted_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def _calculate_relevance_score(self, document: str, query: str) -> float:
        """Calculate relevance score between document and query"""
        try:
            # Simple keyword matching for now
            query_words = set(query.lower().split())
            doc_words = set(document.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))
            
            if union == 0:
                return 0.0
            
            jaccard_similarity = intersection / union
            
            # Consider document length (penalize very long documents)
            length_penalty = min(1.0, 1000 / len(document.split()))
            
            return jaccard_similarity * length_penalty
            
        except:
            return 0.0
    
    def semantic_search_with_context(self, query: str, context: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search with additional context"""
        try:
            # Combine query with context
            enhanced_query = f"{query}\n\nContext: {context}"
            
            # Search with enhanced query
            results = self.search(enhanced_query, n_results)
            
            # Filter results based on context relevance
            filtered_results = []
            for result in results:
                # Check if result is relevant to context
                context_relevance = self._check_context_relevance(result['content'], context)
                if context_relevance > 0.3:  # Threshold
                    result['context_relevance'] = context_relevance
                    filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in semantic search with context: {str(e)}")
            return []
    
    def _check_context_relevance(self, document: str, context: str) -> float:
        """Check relevance of document to context"""
        try:
            # Simple overlap check
            context_words = set(context.lower().split())
            doc_words = set(document.lower().split())
            
            overlap = len(context_words.intersection(doc_words))
            total_context_words = len(context_words)
            
            if total_context_words == 0:
                return 0.0
            
            return overlap / total_context_words
            
        except:
            return 0.0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        try:
            # Get collection count
            count = self.collection.count()
            
            # Get sample documents
            sample_results = self.collection.get(limit=min(5, count))
            
            stats = {
                'collection_name': self.collection_name,
                'document_count': count,
                'sample_documents': sample_results['documents'][:3] if sample_results['documents'] else [],
                'metadata_keys': self._extract_metadata_keys(sample_results.get('metadatas', []))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {'error': str(e)}
    
    def _extract_metadata_keys(self, metadatas: List[Dict[str, Any]]) -> List[str]:
        """Extract unique metadata keys"""
        keys = set()
        for metadata in metadatas:
            if metadata:
                keys.update(metadata.keys())
        return list(keys)
    
    def delete_documents(self, ids: Optional[List[str]] = None, 
                        filter_metadata: Optional[Dict[str, Any]] = None):
        """Delete documents from vector store"""
        try:
            if ids:
                self.collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} documents by ID")
            
            if filter_metadata:
                self.collection.delete(where=filter_metadata)
                logger.info(f"Deleted documents matching filter: {filter_metadata}")
                
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
    
    def clear_collection(self):
        """Clear entire collection"""
        try:
            self.collection.delete(where={})  # Empty filter deletes all
            logger.info(f"Cleared collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
    
    def backup_collection(self, backup_path: str):
        """Backup collection to file"""
        try:
            # Get all data from collection
            all_data = self.collection.get(include=['embeddings', 'documents', 'metadatas'])
            
            # Create backup data structure
            backup_data = {
                'collection_name': self.collection_name,
                'timestamp': datetime.now().isoformat(),
                'document_count': len(all_data['ids']),
                'data': all_data
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Save to file
            with open(backup_path, 'wb') as f:
                pickle.dump(backup_data, f)
            
            logger.info(f"Backed up collection to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up collection: {str(e)}")
            return False
    
    def restore_collection(self, backup_path: str):
        """Restore collection from backup"""
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Load backup data
            with open(backup_path, 'rb') as f:
                backup_data = pickle.load(f)
            
            # Clear existing collection
            self.clear_collection()
            
            # Restore data
            data = backup_data['data']
            if data['ids']:
                self.collection.add(
                    ids=data['ids'],
                    embeddings=data['embeddings'] if 'embeddings' in data else None,
                    documents=data['documents'],
                    metadatas=data['metadatas']
                )
            
            logger.info(f"Restored collection from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring collection: {str(e)}")
            return False