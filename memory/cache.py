"""
Caching layer for performance optimization using Redis via langchain_redis
"""
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import json
import pickle
import os
from loguru import logger
import asyncio
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import Settings

# Import langchain_redis components
from langchain_redis import RedisCache
from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache, get_llm_cache

T = TypeVar('T')

class CacheManager:
    """Manages caching for improved performance using langchain_redis"""
    
    def __init__(self, namespace: str = "default"):
        """Initialize cache manager with langchain_redis"""
        self.settings = Settings()
        self.namespace = namespace
        
        # Initialize Redis cache
        self.redis_cache = None
        self._initialize_redis_cache()
        
        # In-memory cache as fallback
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
    
    def _initialize_redis_cache(self):
        """Initialize Redis cache from langchain_redis if configured"""
        try:
            # Check for Redis URL in environment
            #redis_url = os.getenv('REDIS_URL', "redis://localhost:6379/0")
            #if not redis_url:
            #    redis_url = f"redis://localhost:6379/0"

            self.redis_cache = RedisCache(redis_url="redis://localhost:6379/0")
                
            # Initialize RedisCache with configuration
            # self.redis_cache = RedisCache(
            #     redis_url=redis_url,
            #     # namespace=self.namespace,
            #     # Optional: Configure serializer/deserializer
            #     # serializer=pickle.dumps,
            #     # deserializer=pickle.loads,
            # )
            
            # Test connection
            self.redis_cache.redis.ping()
            logger.info(f"Redis cache initialized with namespace: {self.namespace}")
            
            # Set as global LLM cache (optional, for LangChain LLM caching)
            set_llm_cache(self.redis_cache)
            
        except Exception as e:
            logger.warning(f"Redis cache initialization failed: {str(e)}")
            self.redis_cache = None
    
    def generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        try:
            # Create string representation of arguments
            args_str = str(args)
            kwargs_str = json.dumps(kwargs, sort_keys=True)
            
            # Combine and hash
            key_data = f"{func_name}:{args_str}:{kwargs_str}"
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            
            return key_hash
            
        except Exception as e:
            logger.error(f"Error generating cache key: {str(e)}")
            # Fallback to simple key
            return f"{func_name}:{hash(str(args) + str(kwargs))}"
    
    def _get_full_key(self, key: str) -> str:
        """Get full key with namespace prefix"""
        return f"{self.namespace}:{key}" if self.namespace else key
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            full_key = self._get_full_key(key)
            
            # Try Redis cache first
            if self.redis_cache:
                # Use the underlying redis client for general caching
                cached = self.redis_cache.redis.get(full_key)
                if cached:
                    self.cache_stats['hits'] += 1
                    # Deserialize from bytes
                    return pickle.loads(cached)
            
            # Fallback to memory cache
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if entry['expires'] > datetime.now():
                    self.cache_stats['hits'] += 1
                    return entry['value']
                else:
                    # Expired entry
                    del self.memory_cache[key]
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set value in cache with TTL"""
        try:
            expiration = datetime.now() + timedelta(seconds=ttl_seconds)
            
            # Try Redis cache first
            if self.redis_cache:
                full_key = self._get_full_key(key)
                # Serialize value
                serialized = pickle.dumps(value)
                # Set with expiration
                self.redis_cache.redis.setex(
                    name=full_key,
                    time=ttl_seconds,
                    value=serialized
                )
            else:
                # Fallback to memory cache
                self.memory_cache[key] = {
                    'value': value,
                    'expires': expiration,
                    'created': datetime.now()
                }
                
                # Clean up expired entries occasionally
                if len(self.memory_cache) > 1000:
                    self._cleanup_memory_cache()
            
            self.cache_stats['sets'] += 1
            logger.debug(f"Set cache key: {key}")
            
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
    
    def delete(self, key: str):
        """Delete value from cache"""
        try:
            full_key = self._get_full_key(key)
            
            # Delete from Redis cache
            if self.redis_cache:
                self.redis_cache.redis.delete(full_key)
            
            # Delete from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            self.cache_stats['deletes'] += 1
            logger.debug(f"Deleted cache key: {key}")
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
    
    def clear_namespace(self, pattern: str = None):
        """Clear all keys in namespace"""
        try:
            if pattern is None:
                pattern = f"{self.namespace}:*"
            else:
                pattern = f"{self.namespace}:{pattern}"
            
            # Clear from Redis cache
            if self.redis_cache:
                keys = self.redis_cache.redis.keys(pattern)
                if keys:
                    self.redis_cache.redis.delete(*keys)
            
            # Clear from memory cache
            keys_to_delete = [k for k in self.memory_cache.keys() 
                            if k.startswith(self.namespace)]
            for key in keys_to_delete:
                del self.memory_cache[key]
            
            logger.info(f"Cleared cache namespace: {pattern}")
            
        except Exception as e:
            logger.error(f"Error clearing namespace: {str(e)}")
    
    def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache"""
        try:
            now = datetime.now()
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if entry['expires'] <= now
            ]
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning up memory cache: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache_stats.copy()
        
        # Add cache size info
        if self.redis_cache:
            stats['redis_connected'] = True
            try:
                stats['redis_size'] = self.redis_cache.redis.dbsize()
                
                # Get namespace-specific size
                pattern = f"{self.namespace}:*"
                keys = self.redis_cache.redis.keys(pattern)
                stats['namespace_size'] = len(keys)
            except:
                stats['redis_size'] = 0
                stats['namespace_size'] = 0
        else:
            stats['redis_connected'] = False
            stats['namespace_size'] = 0
        
        stats['memory_cache_size'] = len(self.memory_cache)
        stats['namespace'] = self.namespace
        
        # Calculate hit rate
        total = stats['hits'] + stats['misses']
        stats['hit_rate'] = stats['hits'] / total if total > 0 else 0
        
        return stats
    
    def cache_function(self, ttl_seconds: int = 3600):
        """Decorator to cache function results"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            def wrapper(*args, **kwargs) -> T:
                # Generate cache key
                cache_key = self.generate_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function
                logger.debug(f"Cache miss for {func.__name__}, executing...")
                result = func(*args, **kwargs)
                
                # Cache the result
                self.set(cache_key, result, ttl_seconds)
                
                return result
            
            return wrapper
        
        return decorator
    
    def async_cache_function(self, ttl_seconds: int = 3600):
        """Decorator to cache async function results"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                # Generate cache key
                cache_key = self.generate_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute async function
                logger.debug(f"Cache miss for {func.__name__}, executing...")
                result = await func(*args, **kwargs)
                
                # Cache the result
                self.set(cache_key, result, ttl_seconds)
                
                return result
            
            return wrapper
        
        return decorator
    
    def cache_query(self, query: str, params: Dict[str, Any] = None, ttl_seconds: int = 3600) -> Optional[Any]:
        """Cache database query results"""
        try:
            cache_key = self.generate_key("query", (query,), params or {})
            
            # Try to get from cache
            cached_result = self.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Query cache hit")
                return cached_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error caching query: {str(e)}")
            return None
    
    def set_query_result(self, query: str, result: Any, params: Dict[str, Any] = None, ttl_seconds: int = 3600):
        """Set database query result in cache"""
        try:
            cache_key = self.generate_key("query", (query,), params or {})
            self.set(cache_key, result, ttl_seconds)
            logger.debug(f"Cached query result")
            
        except Exception as e:
            logger.error(f"Error setting query cache: {str(e)}")
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        try:
            full_pattern = f"{self.namespace}:{pattern}" if self.namespace else pattern
            
            if self.redis_cache:
                keys = self.redis_cache.redis.keys(full_pattern)
                if keys:
                    self.redis_cache.redis.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} Redis cache entries for pattern: {pattern}")
            
            # Also check memory cache
            memory_keys = [k for k in self.memory_cache.keys() if pattern in k]
            for key in memory_keys:
                del self.memory_cache[key]
            
            if memory_keys:
                logger.info(f"Invalidated {len(memory_keys)} memory cache entries for pattern: {pattern}")
                
        except Exception as e:
            logger.error(f"Error invalidating pattern: {str(e)}")
    
    # Additional methods for LangChain LLM caching
    
    def cache_llm_generation(self, prompt: str, llm_string: str, generation: str):
        """Cache LLM generation (for use with LangChain)"""
        if self.redis_cache:
            try:
                self.redis_cache.update(prompt, llm_string, [generation])
                logger.debug(f"Cached LLM generation for prompt")
            except Exception as e:
                logger.error(f"Error caching LLM generation: {str(e)}")
    
    def get_llm_generation(self, prompt: str, llm_string: str) -> Optional[str]:
        """Get cached LLM generation (for use with LangChain)"""
        if self.redis_cache:
            try:
                cached = self.redis_cache.lookup(prompt, llm_string)
                if cached:
                    self.cache_stats['hits'] += 1
                    return cached[0] if cached else None
            except Exception as e:
                logger.error(f"Error getting LLM generation: {str(e)}")
        
        self.cache_stats['misses'] += 1
        return None
    
    def clear_llm_cache(self):
        """Clear LLM-specific cache"""
        if self.redis_cache:
            try:
                self.clear_namespace()
                logger.info("Cleared LLM cache")
            except Exception as e:
                logger.error(f"Error clearing LLM cache: {str(e)}")


# Global cache instance for convenience
_global_cache_manager: Optional[CacheManager] = None

def get_cache_manager(namespace: str = "default") -> CacheManager:
    """Get or create a global cache manager instance"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(namespace=namespace)
    return _global_cache_manager


# if __name__ == "__main__":
#     # 1. Initialize manager
#     manager = CacheManager(namespace="test_suite")
    
#     # 2. Check Redis connection
#     if manager.redis_cache:
#         try:
#             response = manager.redis_cache.redis.ping()
#             print(f"Redis Connection Successful: {response}")
#         except Exception as e:
#             print(f"Redis Connection Failed: {e}")
#     else:
#         print("Redis cache is not available.")
    
#     # 3. Test functional caching
#     manager.set("test_key", "Hello Redis with langchain_redis!")
#     print(f"Retrieved: {manager.get('test_key')}")
    
#     # 4. Test LLM caching functionality
#     manager.cache_llm_generation(
#         prompt="What is AI?",
#         llm_string="gpt-3.5-turbo",
#         generation="AI stands for Artificial Intelligence..."
#     )
    
#     cached_gen = manager.get_llm_generation(
#         prompt="What is AI?",
#         llm_string="gpt-3.5-turbo"
#     )
#     print(f"Cached LLM Generation: {cached_gen}")
    
#     # 5. View stats
#     print(f"Cache Stats: {manager.get_stats()}")