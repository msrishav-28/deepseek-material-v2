"""Smart caching system for expensive calculations."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar
import hashlib
import json
import logging
import pickle

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheConfig:
    """Configuration for cache manager."""
    
    cache_dir: Path = Path("./data/cache")
    max_cache_size_mb: int = 1000
    ttl_hours: int = 24
    use_memory_cache: bool = True
    use_disk_cache: bool = True
    
    def __post_init__(self):
        """Create cache directory."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    size_bytes: int = 0
    
    def is_expired(self, ttl_hours: int) -> bool:
        """Check if entry is expired."""
        age = datetime.now() - self.created_at
        return age > timedelta(hours=ttl_hours)
    
    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.now()
        self.access_count += 1


class CacheManager:
    """
    Smart caching system for expensive calculations.
    
    Supports both memory and disk caching with TTL, size limits,
    and LRU eviction policy.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
        }
        
        logger.info(f"Initialized CacheManager at {self.config.cache_dir}")
    
    def _compute_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Compute cache key from function name and arguments."""
        # Create a deterministic string representation
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items())),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        
        # Hash to fixed-length key
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        # Check memory cache first
        if self.config.use_memory_cache and key in self._memory_cache:
            entry = self._memory_cache[key]
            
            # Check if expired
            if entry.is_expired(self.config.ttl_hours):
                del self._memory_cache[key]
                self._stats['misses'] += 1
                return None
            
            # Update access info
            entry.touch()
            self._stats['hits'] += 1
            logger.debug(f"Memory cache hit: {key[:16]}...")
            return entry.value
        
        # Check disk cache
        if self.config.use_disk_cache:
            disk_value = self._get_from_disk(key)
            if disk_value is not None:
                # Promote to memory cache
                if self.config.use_memory_cache:
                    self._put_in_memory(key, disk_value)
                
                self._stats['hits'] += 1
                logger.debug(f"Disk cache hit: {key[:16]}...")
                return disk_value
        
        self._stats['misses'] += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Put in memory cache
        if self.config.use_memory_cache:
            self._put_in_memory(key, value)
        
        # Put in disk cache
        if self.config.use_disk_cache:
            self._put_on_disk(key, value)
    
    def _put_in_memory(self, key: str, value: Any) -> None:
        """Put value in memory cache."""
        # Estimate size
        try:
            size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = 0
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            size_bytes=size_bytes,
        )
        
        self._memory_cache[key] = entry
        
        # Check if we need to evict
        self._evict_if_needed()
    
    def _put_on_disk(self, key: str, value: Any) -> None:
        """Put value in disk cache."""
        cache_file = self.config.cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'value': value,
                    'created_at': datetime.now().isoformat(),
                }, f)
            logger.debug(f"Cached to disk: {key[:16]}...")
        except Exception as e:
            logger.warning(f"Failed to cache to disk: {e}")
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        cache_file = self.config.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if expired
            created_at = datetime.fromisoformat(data['created_at'])
            age = datetime.now() - created_at
            
            if age > timedelta(hours=self.config.ttl_hours):
                cache_file.unlink()
                return None
            
            return data['value']
        
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache size exceeds limit."""
        # Calculate total size
        total_size_mb = sum(
            entry.size_bytes for entry in self._memory_cache.values()
        ) / (1024 * 1024)
        
        if total_size_mb <= self.config.max_cache_size_mb:
            return
        
        # Evict least recently used entries
        entries = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1].accessed_at
        )
        
        while total_size_mb > self.config.max_cache_size_mb * 0.8:  # Evict to 80%
            if not entries:
                break
            
            key, entry = entries.pop(0)
            del self._memory_cache[key]
            total_size_mb -= entry.size_bytes / (1024 * 1024)
            self._stats['evictions'] += 1
            
            logger.debug(f"Evicted cache entry: {key[:16]}...")
    
    def cached(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to cache function results.
        
        Args:
            func: Function to cache
        
        Returns:
            Wrapped function with caching
        """
        def wrapper(*args, **kwargs):
            # Compute cache key
            key = self._compute_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value
            
            # Compute value
            value = func(*args, **kwargs)
            
            # Cache result
            self.put(key, value)
            
            return value
        
        return wrapper
    
    def clear(self) -> None:
        """Clear all caches."""
        # Clear memory cache
        self._memory_cache.clear()
        
        # Clear disk cache
        if self.config.use_disk_cache:
            for cache_file in self.config.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file: {e}")
        
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        memory_size_mb = sum(
            entry.size_bytes for entry in self._memory_cache.values()
        ) / (1024 * 1024)
        
        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'hit_rate': hit_rate,
            'memory_entries': len(self._memory_cache),
            'memory_size_mb': memory_size_mb,
            'disk_entries': len(list(self.config.cache_dir.glob("*.pkl"))),
        }
    
    def print_stats(self) -> None:
        """Print cache statistics."""
        stats = self.get_stats()
        
        print("\n=== Cache Statistics ===")
        print(f"Hits: {stats['hits']}")
        print(f"Misses: {stats['misses']}")
        print(f"Hit Rate: {stats['hit_rate']:.2%}")
        print(f"Evictions: {stats['evictions']}")
        print(f"Memory Entries: {stats['memory_entries']}")
        print(f"Memory Size: {stats['memory_size_mb']:.2f} MB")
        print(f"Disk Entries: {stats['disk_entries']}")
        print("========================\n")
