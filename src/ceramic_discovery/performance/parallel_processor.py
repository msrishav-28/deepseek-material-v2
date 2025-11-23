"""Parallel processing for DFT data collection and computations."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar
import logging
import multiprocessing as mp

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    
    max_workers: Optional[int] = None
    use_processes: bool = True
    chunk_size: int = 1
    timeout_seconds: Optional[float] = None
    
    def __post_init__(self):
        """Set default max_workers based on CPU count."""
        if self.max_workers is None:
            self.max_workers = max(1, mp.cpu_count() - 1)


class ParallelProcessor:
    """
    Parallel processor for DFT data collection and expensive computations.
    
    Supports both process-based and thread-based parallelism with
    configurable workers, chunking, and error handling.
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize parallel processor.
        
        Args:
            config: Parallel processing configuration
        """
        self.config = config or ParallelConfig()
        logger.info(
            f"Initialized ParallelProcessor with {self.config.max_workers} workers "
            f"({'processes' if self.config.use_processes else 'threads'})"
        )
    
    def map(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        show_progress: bool = False
    ) -> List[R]:
        """
        Apply function to items in parallel.
        
        Args:
            func: Function to apply
            items: Items to process
            show_progress: Whether to show progress
        
        Returns:
            List of results
        """
        items_list = list(items)
        total = len(items_list)
        
        if total == 0:
            return []
        
        logger.info(f"Processing {total} items in parallel")
        
        # Choose executor
        executor_class = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor
        
        results = []
        completed = 0
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(func, item): item
                for item in items_list
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_item, timeout=self.config.timeout_seconds):
                item = future_to_item[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if show_progress and completed % max(1, total // 10) == 0:
                        logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")
                
                except Exception as e:
                    logger.error(f"Error processing item {item}: {e}")
                    results.append(None)
                    completed += 1
        
        logger.info(f"Completed processing {completed}/{total} items")
        
        return results
    
    def map_with_errors(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        show_progress: bool = False
    ) -> List[tuple[Optional[R], Optional[str]]]:
        """
        Apply function to items in parallel, returning results and errors.
        
        Args:
            func: Function to apply
            items: Items to process
            show_progress: Whether to show progress
        
        Returns:
            List of (result, error) tuples
        """
        items_list = list(items)
        total = len(items_list)
        
        if total == 0:
            return []
        
        logger.info(f"Processing {total} items in parallel (with error tracking)")
        
        executor_class = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor
        
        results = []
        completed = 0
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            future_to_item = {
                executor.submit(func, item): item
                for item in items_list
            }
            
            for future in as_completed(future_to_item, timeout=self.config.timeout_seconds):
                item = future_to_item[future]
                
                try:
                    result = future.result()
                    results.append((result, None))
                    completed += 1
                    
                    if show_progress and completed % max(1, total // 10) == 0:
                        logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")
                
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"Error processing item {item}: {error_msg}")
                    results.append((None, error_msg))
                    completed += 1
        
        success_count = sum(1 for r, e in results if e is None)
        logger.info(f"Completed: {success_count}/{total} successful, {total-success_count} failed")
        
        return results
    
    def batch_process(
        self,
        func: Callable[[List[T]], R],
        items: Iterable[T],
        batch_size: int,
        show_progress: bool = False
    ) -> List[R]:
        """
        Process items in batches in parallel.
        
        Args:
            func: Function that processes a batch
            items: Items to process
            batch_size: Size of each batch
            show_progress: Whether to show progress
        
        Returns:
            List of batch results
        """
        items_list = list(items)
        
        # Create batches
        batches = [
            items_list[i:i + batch_size]
            for i in range(0, len(items_list), batch_size)
        ]
        
        logger.info(f"Processing {len(items_list)} items in {len(batches)} batches")
        
        return self.map(func, batches, show_progress=show_progress)
    
    def parallel_dft_collection(
        self,
        material_ids: List[str],
        fetch_func: Callable[[str], Any],
        show_progress: bool = True
    ) -> List[tuple[Optional[Any], Optional[str]]]:
        """
        Collect DFT data for multiple materials in parallel.
        
        Args:
            material_ids: List of material IDs to fetch
            fetch_func: Function to fetch data for a single material
            show_progress: Whether to show progress
        
        Returns:
            List of (material_data, error) tuples
        """
        logger.info(f"Starting parallel DFT collection for {len(material_ids)} materials")
        
        return self.map_with_errors(fetch_func, material_ids, show_progress=show_progress)
    
    def parallel_property_calculation(
        self,
        materials: List[Dict[str, Any]],
        calc_func: Callable[[Dict[str, Any]], Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Calculate properties for multiple materials in parallel.
        
        Args:
            materials: List of material dictionaries
            calc_func: Function to calculate properties for a single material
            show_progress: Whether to show progress
        
        Returns:
            List of materials with calculated properties
        """
        logger.info(f"Starting parallel property calculation for {len(materials)} materials")
        
        return self.map(calc_func, materials, show_progress=show_progress)
