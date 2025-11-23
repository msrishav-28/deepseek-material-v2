"""Research progress tracking for screening workflows."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class ScreeningProgress:
    """Progress tracking for screening operations."""
    
    workflow_id: str
    workflow_type: str  # 'dopant_screening', 'batch_screening', 'ml_training'
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Progress metrics
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    
    # Status
    status: str = "running"  # 'running', 'completed', 'failed', 'paused'
    
    # Results summary
    viable_candidates: int = 0
    top_performers: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        processed = self.completed_items + self.failed_items
        if processed == 0:
            return 0.0
        return (self.completed_items / processed) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "progress_percent": self.progress_percent,
            "success_rate": self.success_rate,
            "status": self.status,
            "viable_candidates": self.viable_candidates,
            "top_performers": self.top_performers,
            "metadata": self.metadata
        }


class ProgressTracker:
    """Track research progress across workflows."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize progress tracker.
        
        Args:
            log_dir: Directory to store progress logs
        """
        self.log_dir = log_dir or Path("./logs/progress")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_workflows: Dict[str, ScreeningProgress] = {}
        self.completed_workflows: List[ScreeningProgress] = []
    
    def start_workflow(self, workflow_id: str, workflow_type: str, 
                      total_items: int, metadata: Optional[Dict[str, Any]] = None) -> ScreeningProgress:
        """Start tracking a new workflow.
        
        Args:
            workflow_id: Unique identifier for the workflow
            workflow_type: Type of workflow
            total_items: Total number of items to process
            metadata: Additional metadata
            
        Returns:
            Progress tracking object
        """
        progress = ScreeningProgress(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            start_time=datetime.now(),
            total_items=total_items,
            metadata=metadata or {}
        )
        
        self.active_workflows[workflow_id] = progress
        self._save_progress(progress)
        
        return progress
    
    def update_progress(self, workflow_id: str, completed: int = 0, failed: int = 0, 
                       skipped: int = 0, viable: int = 0) -> ScreeningProgress:
        """Update workflow progress.
        
        Args:
            workflow_id: Workflow identifier
            completed: Number of newly completed items
            failed: Number of newly failed items
            skipped: Number of newly skipped items
            viable: Number of newly identified viable candidates
            
        Returns:
            Updated progress object
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        progress = self.active_workflows[workflow_id]
        progress.completed_items += completed
        progress.failed_items += failed
        progress.skipped_items += skipped
        progress.viable_candidates += viable
        
        self._save_progress(progress)
        
        return progress
    
    def complete_workflow(self, workflow_id: str, top_performers: Optional[List[str]] = None) -> ScreeningProgress:
        """Mark workflow as completed.
        
        Args:
            workflow_id: Workflow identifier
            top_performers: List of top performing candidates
            
        Returns:
            Completed progress object
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        progress = self.active_workflows[workflow_id]
        progress.end_time = datetime.now()
        progress.status = "completed"
        
        if top_performers:
            progress.top_performers = top_performers
        
        self._save_progress(progress)
        
        # Move to completed
        self.completed_workflows.append(progress)
        del self.active_workflows[workflow_id]
        
        return progress
    
    def fail_workflow(self, workflow_id: str, error: str) -> ScreeningProgress:
        """Mark workflow as failed.
        
        Args:
            workflow_id: Workflow identifier
            error: Error message
            
        Returns:
            Failed progress object
        """
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        progress = self.active_workflows[workflow_id]
        progress.end_time = datetime.now()
        progress.status = "failed"
        progress.metadata["error"] = error
        
        self._save_progress(progress)
        
        # Move to completed
        self.completed_workflows.append(progress)
        del self.active_workflows[workflow_id]
        
        return progress
    
    def get_progress(self, workflow_id: str) -> Optional[ScreeningProgress]:
        """Get progress for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Progress object or None if not found
        """
        return self.active_workflows.get(workflow_id)
    
    def get_all_active(self) -> List[ScreeningProgress]:
        """Get all active workflows.
        
        Returns:
            List of active progress objects
        """
        return list(self.active_workflows.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all workflows.
        
        Returns:
            Summary statistics
        """
        all_workflows = list(self.active_workflows.values()) + self.completed_workflows
        
        if not all_workflows:
            return {"message": "No workflows tracked yet"}
        
        total_items = sum(w.total_items for w in all_workflows)
        total_completed = sum(w.completed_items for w in all_workflows)
        total_viable = sum(w.viable_candidates for w in all_workflows)
        
        return {
            "total_workflows": len(all_workflows),
            "active_workflows": len(self.active_workflows),
            "completed_workflows": len(self.completed_workflows),
            "total_items_processed": total_completed,
            "total_items_planned": total_items,
            "total_viable_candidates": total_viable,
            "workflows": [w.to_dict() for w in all_workflows]
        }
    
    def _save_progress(self, progress: ScreeningProgress) -> None:
        """Save progress to file.
        
        Args:
            progress: Progress object to save
        """
        progress_file = self.log_dir / f"{progress.workflow_id}.json"
        
        with open(progress_file, "w") as f:
            json.dump(progress.to_dict(), f, indent=2)
    
    def load_workflow(self, workflow_id: str) -> Optional[ScreeningProgress]:
        """Load workflow progress from file.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Progress object or None if not found
        """
        progress_file = self.log_dir / f"{workflow_id}.json"
        
        if not progress_file.exists():
            return None
        
        with open(progress_file) as f:
            data = json.load(f)
        
        progress = ScreeningProgress(
            workflow_id=data["workflow_id"],
            workflow_type=data["workflow_type"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
            total_items=data["total_items"],
            completed_items=data["completed_items"],
            failed_items=data["failed_items"],
            skipped_items=data["skipped_items"],
            status=data["status"],
            viable_candidates=data["viable_candidates"],
            top_performers=data["top_performers"],
            metadata=data["metadata"]
        )
        
        return progress
