"""Workflow orchestrator for high-throughput screening using Prefect."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import logging
import time

try:
    from prefect import flow, task, get_run_logger
    from prefect.task_runners import ConcurrentTaskRunner
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    import warnings
    warnings.warn("Prefect not available. Install with: pip install prefect")


logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Status of individual task execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of a task execution."""
    
    task_name: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_name': self.task_name,
            'status': self.status.value,
            'result': str(self.result) if self.result is not None else None,
            'error': self.error,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'retry_count': self.retry_count,
        }


@dataclass
class WorkflowProgress:
    """Progress tracking for workflow execution."""
    
    workflow_id: str
    status: WorkflowStatus
    total_tasks: int
    completed_tasks: int = 0
    failed_tasks: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    task_results: List[TaskResult] = field(default_factory=list)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'workflow_id': self.workflow_id,
            'status': self.status.value,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'progress_percentage': self.progress_percentage,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'task_results': [tr.to_dict() for tr in self.task_results],
        }


@dataclass
class RetryConfig:
    """Configuration for task retry logic."""
    
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    max_delay_seconds: float = 60.0
    
    def get_delay(self, retry_count: int) -> float:
        """Calculate delay for retry with exponential backoff."""
        delay = self.initial_delay_seconds * (self.backoff_factor ** retry_count)
        return min(delay, self.max_delay_seconds)


class WorkflowOrchestrator:
    """
    Orchestrator for high-throughput screening workflows.
    
    Manages workflow execution, job scheduling, progress tracking,
    and error handling with retry mechanisms.
    """
    
    def __init__(
        self,
        workflow_id: str,
        max_parallel_jobs: int = 4,
        retry_config: Optional[RetryConfig] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize workflow orchestrator.
        
        Args:
            workflow_id: Unique identifier for workflow
            max_parallel_jobs: Maximum number of parallel jobs
            retry_config: Configuration for retry logic
            checkpoint_dir: Directory for saving checkpoints
        """
        self.workflow_id = workflow_id
        self.max_parallel_jobs = max_parallel_jobs
        self.retry_config = retry_config or RetryConfig()
        self.checkpoint_dir = checkpoint_dir or Path("./results/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress: Optional[WorkflowProgress] = None
        self._task_registry: Dict[str, Callable] = {}
        
        if not PREFECT_AVAILABLE:
            logger.warning("Prefect not available. Using fallback sequential execution.")
    
    def register_task(self, name: str, func: Callable) -> None:
        """
        Register a task function.
        
        Args:
            name: Task name
            func: Task function
        """
        self._task_registry[name] = func
        logger.info(f"Registered task: {name}")
    
    def execute_workflow(
        self,
        tasks: List[Dict[str, Any]],
        use_parallel: bool = True
    ) -> WorkflowProgress:
        """
        Execute workflow with registered tasks.
        
        Args:
            tasks: List of task configurations
            use_parallel: Whether to use parallel execution
        
        Returns:
            Workflow progress with results
        """
        # Initialize progress tracking
        self.progress = WorkflowProgress(
            workflow_id=self.workflow_id,
            status=WorkflowStatus.RUNNING,
            total_tasks=len(tasks),
            start_time=datetime.now()
        )
        
        logger.info(f"Starting workflow {self.workflow_id} with {len(tasks)} tasks")
        
        try:
            if PREFECT_AVAILABLE and use_parallel:
                results = self._execute_with_prefect(tasks)
            else:
                results = self._execute_sequential(tasks)
            
            # Update final status
            self.progress.status = WorkflowStatus.COMPLETED
            self.progress.end_time = datetime.now()
            
            logger.info(
                f"Workflow {self.workflow_id} completed. "
                f"Success: {self.progress.completed_tasks}/{self.progress.total_tasks}"
            )
            
        except Exception as e:
            self.progress.status = WorkflowStatus.FAILED
            self.progress.end_time = datetime.now()
            logger.error(f"Workflow {self.workflow_id} failed: {e}")
            raise
        
        finally:
            # Save checkpoint
            self._save_checkpoint()
        
        return self.progress
    
    def _execute_sequential(self, tasks: List[Dict[str, Any]]) -> List[TaskResult]:
        """Execute tasks sequentially."""
        results = []
        
        for task_config in tasks:
            result = self._execute_task_with_retry(task_config)
            results.append(result)
            self.progress.task_results.append(result)
            
            if result.status == TaskStatus.COMPLETED:
                self.progress.completed_tasks += 1
            elif result.status == TaskStatus.FAILED:
                self.progress.failed_tasks += 1
            
            # Save checkpoint after each task
            self._save_checkpoint()
        
        return results

    def _execute_with_prefect(self, tasks: List[Dict[str, Any]]) -> List[TaskResult]:
        """Execute tasks using Prefect for parallel execution."""
        
        @flow(name=f"workflow_{self.workflow_id}", task_runner=ConcurrentTaskRunner())
        def workflow_flow():
            """Prefect flow for parallel task execution."""
            task_futures = []
            
            for task_config in tasks:
                future = self._create_prefect_task(task_config)
                task_futures.append(future)
            
            return task_futures
        
        # Execute flow
        results = workflow_flow()
        
        # Process results
        for result in results:
            self.progress.task_results.append(result)
            
            if result.status == TaskStatus.COMPLETED:
                self.progress.completed_tasks += 1
            elif result.status == TaskStatus.FAILED:
                self.progress.failed_tasks += 1
        
        return results
    
    def _create_prefect_task(self, task_config: Dict[str, Any]) -> TaskResult:
        """Create and execute a Prefect task."""
        
        @task(name=task_config['name'], retries=self.retry_config.max_retries)
        def prefect_task():
            """Prefect task wrapper."""
            return self._execute_task_with_retry(task_config)
        
        return prefect_task()
    
    def _execute_task_with_retry(self, task_config: Dict[str, Any]) -> TaskResult:
        """
        Execute a task with retry logic.
        
        Args:
            task_config: Task configuration dictionary
        
        Returns:
            Task result
        """
        task_name = task_config['name']
        task_func = self._task_registry.get(task_name)
        
        if task_func is None:
            return TaskResult(
                task_name=task_name,
                status=TaskStatus.FAILED,
                error=f"Task function not registered: {task_name}"
            )
        
        task_args = task_config.get('args', [])
        task_kwargs = task_config.get('kwargs', {})
        
        result = TaskResult(
            task_name=task_name,
            status=TaskStatus.RUNNING,
            start_time=datetime.now()
        )
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.retry_config.max_retries:
            try:
                logger.info(f"Executing task: {task_name} (attempt {retry_count + 1})")
                
                # Execute task
                task_result = task_func(*task_args, **task_kwargs)
                
                # Success
                result.status = TaskStatus.COMPLETED
                result.result = task_result
                result.end_time = datetime.now()
                result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                result.retry_count = retry_count
                
                logger.info(f"Task {task_name} completed successfully")
                return result
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Task {task_name} failed (attempt {retry_count + 1}): {e}")
                
                retry_count += 1
                
                if retry_count <= self.retry_config.max_retries:
                    # Calculate delay and retry
                    delay = self.retry_config.get_delay(retry_count - 1)
                    logger.info(f"Retrying task {task_name} in {delay:.1f} seconds...")
                    result.status = TaskStatus.RETRYING
                    time.sleep(delay)
                else:
                    # Max retries exceeded
                    result.status = TaskStatus.FAILED
                    result.error = last_error
                    result.end_time = datetime.now()
                    result.duration_seconds = (result.end_time - result.start_time).total_seconds()
                    result.retry_count = retry_count - 1
                    
                    logger.error(f"Task {task_name} failed after {retry_count} attempts")
                    return result
        
        return result
    
    def _save_checkpoint(self) -> None:
        """Save workflow progress checkpoint."""
        if self.progress is None:
            return
        
        checkpoint_file = self.checkpoint_dir / f"{self.workflow_id}_checkpoint.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(self.progress.to_dict(), f, indent=2)
            
            logger.debug(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self) -> Optional[WorkflowProgress]:
        """
        Load workflow progress from checkpoint.
        
        Returns:
            Workflow progress if checkpoint exists, None otherwise
        """
        checkpoint_file = self.checkpoint_dir / f"{self.workflow_id}_checkpoint.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct progress
            progress = WorkflowProgress(
                workflow_id=data['workflow_id'],
                status=WorkflowStatus(data['status']),
                total_tasks=data['total_tasks'],
                completed_tasks=data['completed_tasks'],
                failed_tasks=data['failed_tasks'],
                start_time=datetime.fromisoformat(data['start_time']) if data['start_time'] else None,
                end_time=datetime.fromisoformat(data['end_time']) if data['end_time'] else None,
            )
            
            # Reconstruct task results
            for tr_data in data['task_results']:
                task_result = TaskResult(
                    task_name=tr_data['task_name'],
                    status=TaskStatus(tr_data['status']),
                    result=tr_data['result'],
                    error=tr_data['error'],
                    start_time=datetime.fromisoformat(tr_data['start_time']) if tr_data['start_time'] else None,
                    end_time=datetime.fromisoformat(tr_data['end_time']) if tr_data['end_time'] else None,
                    duration_seconds=tr_data['duration_seconds'],
                    retry_count=tr_data['retry_count'],
                )
                progress.task_results.append(task_result)
            
            logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return progress
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def resume_workflow(self, tasks: List[Dict[str, Any]]) -> WorkflowProgress:
        """
        Resume workflow from checkpoint.
        
        Args:
            tasks: List of task configurations
        
        Returns:
            Workflow progress with results
        """
        # Load checkpoint
        self.progress = self.load_checkpoint()
        
        if self.progress is None:
            logger.info("No checkpoint found. Starting new workflow.")
            return self.execute_workflow(tasks)
        
        logger.info(
            f"Resuming workflow {self.workflow_id} from checkpoint. "
            f"Progress: {self.progress.completed_tasks}/{self.progress.total_tasks}"
        )
        
        # Filter out completed tasks
        completed_task_names = {
            tr.task_name for tr in self.progress.task_results
            if tr.status == TaskStatus.COMPLETED
        }
        
        remaining_tasks = [
            task for task in tasks
            if task['name'] not in completed_task_names
        ]
        
        if not remaining_tasks:
            logger.info("All tasks already completed.")
            self.progress.status = WorkflowStatus.COMPLETED
            return self.progress
        
        logger.info(f"Resuming with {len(remaining_tasks)} remaining tasks")
        
        # Update progress
        self.progress.status = WorkflowStatus.RUNNING
        
        # Execute remaining tasks
        try:
            results = self._execute_sequential(remaining_tasks)
            
            self.progress.status = WorkflowStatus.COMPLETED
            self.progress.end_time = datetime.now()
            
        except Exception as e:
            self.progress.status = WorkflowStatus.FAILED
            self.progress.end_time = datetime.now()
            logger.error(f"Workflow resume failed: {e}")
            raise
        
        finally:
            self._save_checkpoint()
        
        return self.progress
    
    def cancel_workflow(self) -> None:
        """Cancel running workflow."""
        if self.progress:
            self.progress.status = WorkflowStatus.CANCELLED
            self.progress.end_time = datetime.now()
            self._save_checkpoint()
            logger.info(f"Workflow {self.workflow_id} cancelled")
    
    def get_progress(self) -> Optional[WorkflowProgress]:
        """Get current workflow progress."""
        return self.progress
    
    def get_task_result(self, task_name: str) -> Optional[TaskResult]:
        """
        Get result for a specific task.
        
        Args:
            task_name: Name of task
        
        Returns:
            Task result if found, None otherwise
        """
        if self.progress is None:
            return None
        
        for task_result in self.progress.task_results:
            if task_result.task_name == task_name:
                return task_result
        
        return None


class JobScheduler:
    """
    Job scheduler for managing computational resources.
    
    Schedules jobs based on resource availability and priority.
    """
    
    def __init__(
        self,
        max_concurrent_jobs: int = 4,
        resource_limits: Optional[Dict[str, int]] = None
    ):
        """
        Initialize job scheduler.
        
        Args:
            max_concurrent_jobs: Maximum number of concurrent jobs
            resource_limits: Optional resource limits (e.g., {'cpu': 16, 'memory_gb': 64})
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.resource_limits = resource_limits or {}
        
        self.active_jobs: List[Dict[str, Any]] = []
        self.queued_jobs: List[Dict[str, Any]] = []
        self.completed_jobs: List[Dict[str, Any]] = []
    
    def submit_job(
        self,
        job_id: str,
        job_func: Callable,
        priority: int = 0,
        resource_requirements: Optional[Dict[str, int]] = None
    ) -> str:
        """
        Submit a job to the scheduler.
        
        Args:
            job_id: Unique job identifier
            job_func: Job function to execute
            priority: Job priority (higher = more important)
            resource_requirements: Required resources
        
        Returns:
            Job ID
        """
        job = {
            'job_id': job_id,
            'job_func': job_func,
            'priority': priority,
            'resource_requirements': resource_requirements or {},
            'status': 'queued',
            'submit_time': datetime.now(),
        }
        
        self.queued_jobs.append(job)
        self.queued_jobs.sort(key=lambda x: x['priority'], reverse=True)
        
        logger.info(f"Job {job_id} submitted with priority {priority}")
        
        return job_id
    
    def can_schedule_job(self, job: Dict[str, Any]) -> bool:
        """
        Check if job can be scheduled based on resource availability.
        
        Args:
            job: Job dictionary
        
        Returns:
            True if job can be scheduled
        """
        # Check concurrent job limit
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            return False
        
        # Check resource limits
        for resource, required in job['resource_requirements'].items():
            if resource in self.resource_limits:
                # Calculate current usage
                current_usage = sum(
                    j['resource_requirements'].get(resource, 0)
                    for j in self.active_jobs
                )
                
                if current_usage + required > self.resource_limits[resource]:
                    return False
        
        return True
    
    def schedule_jobs(self) -> List[str]:
        """
        Schedule queued jobs based on resource availability.
        
        Returns:
            List of scheduled job IDs
        """
        scheduled = []
        
        for job in list(self.queued_jobs):
            if self.can_schedule_job(job):
                # Move to active jobs
                self.queued_jobs.remove(job)
                job['status'] = 'running'
                job['start_time'] = datetime.now()
                self.active_jobs.append(job)
                
                scheduled.append(job['job_id'])
                logger.info(f"Job {job['job_id']} scheduled")
        
        return scheduled
    
    def complete_job(self, job_id: str, result: Any = None, error: Optional[str] = None) -> None:
        """
        Mark job as completed.
        
        Args:
            job_id: Job identifier
            result: Job result
            error: Error message if job failed
        """
        # Find job in active jobs
        job = None
        for j in self.active_jobs:
            if j['job_id'] == job_id:
                job = j
                break
        
        if job is None:
            logger.warning(f"Job {job_id} not found in active jobs")
            return
        
        # Move to completed jobs
        self.active_jobs.remove(job)
        job['status'] = 'completed' if error is None else 'failed'
        job['end_time'] = datetime.now()
        job['result'] = result
        job['error'] = error
        self.completed_jobs.append(job)
        
        logger.info(f"Job {job_id} completed with status: {job['status']}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            'active_jobs': len(self.active_jobs),
            'queued_jobs': len(self.queued_jobs),
            'completed_jobs': len(self.completed_jobs),
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'resource_limits': self.resource_limits,
        }
