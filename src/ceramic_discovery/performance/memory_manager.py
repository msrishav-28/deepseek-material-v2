"""Memory-efficient data handling for large datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, List, Optional, TypeVar
import logging

import numpy as np
import pandas as pd

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    import warnings
    warnings.warn("h5py not available. Install with: pip install h5py")

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ChunkConfig:
    """Configuration for chunk processing."""
    
    chunk_size: int = 1000
    overlap: int = 0
    use_hdf5: bool = True
    hdf5_compression: str = 'gzip'


class ChunkProcessor:
    """
    Processor for handling data in chunks.
    
    Enables memory-efficient processing of large datasets by
    loading and processing data in manageable chunks.
    """
    
    def __init__(self, chunk_size: int = 1000):
        """
        Initialize chunk processor.
        
        Args:
            chunk_size: Size of each chunk
        """
        self.chunk_size = chunk_size
        logger.info(f"Initialized ChunkProcessor with chunk_size={chunk_size}")
    
    def process_dataframe_chunks(
        self,
        df: pd.DataFrame,
        process_func: Callable[[pd.DataFrame], pd.DataFrame],
        show_progress: bool = False
    ) -> pd.DataFrame:
        """
        Process DataFrame in chunks.
        
        Args:
            df: DataFrame to process
            process_func: Function to apply to each chunk
            show_progress: Whether to show progress
        
        Returns:
            Processed DataFrame
        """
        total_rows = len(df)
        num_chunks = (total_rows + self.chunk_size - 1) // self.chunk_size
        
        logger.info(f"Processing {total_rows} rows in {num_chunks} chunks")
        
        results = []
        
        for i in range(0, total_rows, self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size]
            processed_chunk = process_func(chunk)
            results.append(processed_chunk)
            
            if show_progress and (i // self.chunk_size) % max(1, num_chunks // 10) == 0:
                progress = (i + len(chunk)) / total_rows * 100
                logger.info(f"Progress: {progress:.1f}%")
        
        return pd.concat(results, ignore_index=True)
    
    def process_file_chunks(
        self,
        file_path: Path,
        process_func: Callable[[pd.DataFrame], Any],
        file_format: str = 'csv',
        show_progress: bool = False
    ) -> List[Any]:
        """
        Process file in chunks without loading entire file into memory.
        
        Args:
            file_path: Path to file
            process_func: Function to apply to each chunk
            file_format: File format ('csv', 'parquet', 'json')
            show_progress: Whether to show progress
        
        Returns:
            List of results from each chunk
        """
        logger.info(f"Processing file {file_path} in chunks")
        
        results = []
        chunk_num = 0
        
        if file_format == 'csv':
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                result = process_func(chunk)
                results.append(result)
                chunk_num += 1
                
                if show_progress and chunk_num % 10 == 0:
                    logger.info(f"Processed {chunk_num} chunks")
        
        elif file_format == 'parquet':
            # Parquet supports efficient chunked reading
            df = pd.read_parquet(file_path)
            for i in range(0, len(df), self.chunk_size):
                chunk = df.iloc[i:i + self.chunk_size]
                result = process_func(chunk)
                results.append(result)
                chunk_num += 1
                
                if show_progress and chunk_num % 10 == 0:
                    logger.info(f"Processed {chunk_num} chunks")
        
        elif file_format == 'json':
            for chunk in pd.read_json(file_path, lines=True, chunksize=self.chunk_size):
                result = process_func(chunk)
                results.append(result)
                chunk_num += 1
                
                if show_progress and chunk_num % 10 == 0:
                    logger.info(f"Processed {chunk_num} chunks")
        
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Processed {chunk_num} chunks total")
        
        return results
    
    def aggregate_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        agg_func: Callable[[List[pd.DataFrame]], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Aggregate processed chunks.
        
        Args:
            chunks: Iterable of DataFrame chunks
            agg_func: Function to aggregate chunks
        
        Returns:
            Aggregated DataFrame
        """
        chunk_list = list(chunks)
        logger.info(f"Aggregating {len(chunk_list)} chunks")
        
        return agg_func(chunk_list)


class MemoryManager:
    """
    Memory-efficient data handling for large datasets.
    
    Provides utilities for streaming data processing, HDF5 storage,
    and memory usage monitoring.
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        Initialize memory manager.
        
        Args:
            config: Chunk processing configuration
        """
        self.config = config or ChunkConfig()
        self.chunk_processor = ChunkProcessor(chunk_size=self.config.chunk_size)
        
        logger.info("Initialized MemoryManager")
    
    def stream_process(
        self,
        data_source: Iterable[T],
        process_func: Callable[[T], Any],
        batch_size: int = 100
    ) -> Generator[Any, None, None]:
        """
        Stream process data without loading all into memory.
        
        Args:
            data_source: Iterable data source
            process_func: Function to process each item
            batch_size: Size of processing batches
        
        Yields:
            Processed results
        """
        batch = []
        
        for item in data_source:
            batch.append(item)
            
            if len(batch) >= batch_size:
                # Process batch
                for b_item in batch:
                    yield process_func(b_item)
                batch = []
        
        # Process remaining items
        for b_item in batch:
            yield process_func(b_item)
    
    def save_to_hdf5(
        self,
        data: pd.DataFrame,
        file_path: Path,
        dataset_name: str = 'data',
        compression: Optional[str] = None
    ) -> None:
        """
        Save DataFrame to HDF5 format for efficient storage.
        
        Args:
            data: DataFrame to save
            file_path: Output file path
            dataset_name: Name of dataset in HDF5 file
            compression: Compression algorithm ('gzip', 'lzf', None)
        """
        if not HDF5_AVAILABLE:
            logger.warning("HDF5 not available. Falling back to parquet format.")
            data.to_parquet(file_path.with_suffix('.parquet'), compression='gzip')
            return
        
        compression = compression or self.config.hdf5_compression
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to HDF5
        data.to_hdf(
            file_path,
            key=dataset_name,
            mode='w',
            complevel=9 if compression else 0,
            complib=compression
        )
        
        logger.info(f"Saved {len(data)} rows to HDF5: {file_path}")
    
    def load_from_hdf5(
        self,
        file_path: Path,
        dataset_name: str = 'data',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load DataFrame from HDF5 format.
        
        Args:
            file_path: Input file path
            dataset_name: Name of dataset in HDF5 file
            columns: Optional list of columns to load
        
        Returns:
            Loaded DataFrame
        """
        if not HDF5_AVAILABLE:
            logger.warning("HDF5 not available. Trying parquet format.")
            parquet_path = file_path.with_suffix('.parquet')
            if parquet_path.exists():
                return pd.read_parquet(parquet_path, columns=columns)
            raise FileNotFoundError(f"Neither HDF5 nor parquet file found: {file_path}")
        
        # Load from HDF5
        df = pd.read_hdf(file_path, key=dataset_name, columns=columns)
        
        logger.info(f"Loaded {len(df)} rows from HDF5: {file_path}")
        
        return df
    
    def load_hdf5_chunks(
        self,
        file_path: Path,
        dataset_name: str = 'data',
        chunk_size: Optional[int] = None
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Load HDF5 file in chunks.
        
        Args:
            file_path: Input file path
            dataset_name: Name of dataset in HDF5 file
            chunk_size: Size of chunks (uses config default if None)
        
        Yields:
            DataFrame chunks
        """
        if not HDF5_AVAILABLE:
            raise ImportError("HDF5 not available. Install with: pip install h5py")
        
        chunk_size = chunk_size or self.config.chunk_size
        
        # Use iterator for chunked reading
        for chunk in pd.read_hdf(file_path, key=dataset_name, chunksize=chunk_size):
            yield chunk
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by downcasting numeric types.
        
        Args:
            df: DataFrame to optimize
        
        Returns:
            Optimized DataFrame
        """
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to category if beneficial
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (start_mem - end_mem) / start_mem * 100
        
        logger.info(
            f"Memory optimization: {start_mem:.2f} MB -> {end_mem:.2f} MB "
            f"({reduction:.1f}% reduction)"
        )
        
        return df
    
    def get_memory_usage(self, obj: Any) -> float:
        """
        Get memory usage of object in MB.
        
        Args:
            obj: Object to measure
        
        Returns:
            Memory usage in MB
        """
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum() / 1024**2
        elif isinstance(obj, np.ndarray):
            return obj.nbytes / 1024**2
        else:
            import sys
            return sys.getsizeof(obj) / 1024**2
