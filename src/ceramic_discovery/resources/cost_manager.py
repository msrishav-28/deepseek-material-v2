"""Cost-aware job scheduling for external APIs and computational resources."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of computational resources."""
    
    API_CALL = "api_call"
    CPU_HOUR = "cpu_hour"
    GPU_HOUR = "gpu_hour"
    MEMORY_GB_HOUR = "memory_gb_hour"
    STORAGE_GB = "storage_gb"
    NETWORK_GB = "network_gb"


@dataclass
class ResourceCost:
    """Cost information for a resource."""
    
    resource_type: ResourceType
    cost_per_unit: float
    currency: str = "USD"
    rate_limit_per_hour: Optional[int] = None
    rate_limit_per_day: Optional[int] = None
    
    def calculate_cost(self, units: float) -> float:
        """Calculate cost for given units."""
        return units * self.cost_per_unit


@dataclass
class CostConfig:
    """Configuration for cost management."""
    
    # API costs
    materials_project_cost_per_call: float = 0.0  # Free but rate-limited
    materials_project_rate_limit_per_hour: int = 1000
    
    # Compute costs (example values)
    cpu_cost_per_hour: float = 0.10
    gpu_cost_per_hour: float = 1.00
    memory_cost_per_gb_hour: float = 0.01
    storage_cost_per_gb: float = 0.02
    
    # Budget limits
    daily_budget: float = 100.0
    monthly_budget: float = 2000.0
    
    # Alert thresholds
    budget_warning_threshold: float = 0.8  # 80%
    budget_critical_threshold: float = 0.95  # 95%


@dataclass
class ResourceUsageRecord:
    """Record of resource usage."""
    
    resource_type: ResourceType
    units: float
    cost: float
    timestamp: datetime
    job_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class CostManager:
    """
    Cost-aware job scheduling for external APIs and computational resources.
    
    Tracks resource usage, enforces rate limits, and manages budgets
    to prevent unexpected costs.
    """
    
    def __init__(self, config: Optional[CostConfig] = None):
        """
        Initialize cost manager.
        
        Args:
            config: Cost configuration
        """
        self.config = config or CostConfig()
        
        # Define resource costs
        self.resource_costs = {
            ResourceType.API_CALL: ResourceCost(
                resource_type=ResourceType.API_CALL,
                cost_per_unit=self.config.materials_project_cost_per_call,
                rate_limit_per_hour=self.config.materials_project_rate_limit_per_hour,
            ),
            ResourceType.CPU_HOUR: ResourceCost(
                resource_type=ResourceType.CPU_HOUR,
                cost_per_unit=self.config.cpu_cost_per_hour,
            ),
            ResourceType.GPU_HOUR: ResourceCost(
                resource_type=ResourceType.GPU_HOUR,
                cost_per_unit=self.config.gpu_cost_per_hour,
            ),
            ResourceType.MEMORY_GB_HOUR: ResourceCost(
                resource_type=ResourceType.MEMORY_GB_HOUR,
                cost_per_unit=self.config.memory_cost_per_gb_hour,
            ),
            ResourceType.STORAGE_GB: ResourceCost(
                resource_type=ResourceType.STORAGE_GB,
                cost_per_unit=self.config.storage_cost_per_gb,
            ),
        }
        
        # Usage tracking
        self.usage_records: List[ResourceUsageRecord] = []
        self._api_call_timestamps: List[datetime] = []
        
        logger.info("Initialized CostManager")
    
    def can_make_api_call(self) -> bool:
        """
        Check if API call can be made within rate limits.
        
        Returns:
            True if call can be made
        """
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old timestamps
        self._api_call_timestamps = [
            ts for ts in self._api_call_timestamps
            if ts > hour_ago
        ]
        
        # Check rate limit
        rate_limit = self.resource_costs[ResourceType.API_CALL].rate_limit_per_hour
        
        if rate_limit and len(self._api_call_timestamps) >= rate_limit:
            logger.warning(f"API rate limit reached: {rate_limit} calls/hour")
            return False
        
        return True
    
    def record_api_call(self, job_id: Optional[str] = None) -> None:
        """
        Record an API call.
        
        Args:
            job_id: Optional job identifier
        """
        now = datetime.now()
        self._api_call_timestamps.append(now)
        
        cost = self.resource_costs[ResourceType.API_CALL].cost_per_unit
        
        record = ResourceUsageRecord(
            resource_type=ResourceType.API_CALL,
            units=1.0,
            cost=cost,
            timestamp=now,
            job_id=job_id,
        )
        
        self.usage_records.append(record)
        logger.debug(f"Recorded API call (cost: ${cost:.4f})")
    
    def record_compute_usage(
        self,
        cpu_hours: float = 0.0,
        gpu_hours: float = 0.0,
        memory_gb_hours: float = 0.0,
        job_id: Optional[str] = None
    ) -> float:
        """
        Record computational resource usage.
        
        Args:
            cpu_hours: CPU hours used
            gpu_hours: GPU hours used
            memory_gb_hours: Memory GB-hours used
            job_id: Optional job identifier
        
        Returns:
            Total cost
        """
        now = datetime.now()
        total_cost = 0.0
        
        if cpu_hours > 0:
            cost = self.resource_costs[ResourceType.CPU_HOUR].calculate_cost(cpu_hours)
            record = ResourceUsageRecord(
                resource_type=ResourceType.CPU_HOUR,
                units=cpu_hours,
                cost=cost,
                timestamp=now,
                job_id=job_id,
            )
            self.usage_records.append(record)
            total_cost += cost
        
        if gpu_hours > 0:
            cost = self.resource_costs[ResourceType.GPU_HOUR].calculate_cost(gpu_hours)
            record = ResourceUsageRecord(
                resource_type=ResourceType.GPU_HOUR,
                units=gpu_hours,
                cost=cost,
                timestamp=now,
                job_id=job_id,
            )
            self.usage_records.append(record)
            total_cost += cost
        
        if memory_gb_hours > 0:
            cost = self.resource_costs[ResourceType.MEMORY_GB_HOUR].calculate_cost(memory_gb_hours)
            record = ResourceUsageRecord(
                resource_type=ResourceType.MEMORY_GB_HOUR,
                units=memory_gb_hours,
                cost=cost,
                timestamp=now,
                job_id=job_id,
            )
            self.usage_records.append(record)
            total_cost += cost
        
        logger.info(f"Recorded compute usage (cost: ${total_cost:.2f})")
        
        return total_cost
    
    def get_daily_cost(self) -> float:
        """Get total cost for current day."""
        now = datetime.now()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        daily_records = [
            record for record in self.usage_records
            if record.timestamp >= day_start
        ]
        
        return sum(record.cost for record in daily_records)
    
    def get_monthly_cost(self) -> float:
        """Get total cost for current month."""
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        monthly_records = [
            record for record in self.usage_records
            if record.timestamp >= month_start
        ]
        
        return sum(record.cost for record in monthly_records)
    
    def check_budget_status(self) -> Dict[str, any]:
        """
        Check budget status and return warnings if needed.
        
        Returns:
            Dictionary with budget status
        """
        daily_cost = self.get_daily_cost()
        monthly_cost = self.get_monthly_cost()
        
        daily_usage_pct = daily_cost / self.config.daily_budget
        monthly_usage_pct = monthly_cost / self.config.monthly_budget
        
        status = {
            'daily_cost': daily_cost,
            'daily_budget': self.config.daily_budget,
            'daily_usage_pct': daily_usage_pct,
            'monthly_cost': monthly_cost,
            'monthly_budget': self.config.monthly_budget,
            'monthly_usage_pct': monthly_usage_pct,
            'warnings': [],
        }
        
        # Check for warnings
        if daily_usage_pct >= self.config.budget_critical_threshold:
            status['warnings'].append(
                f"CRITICAL: Daily budget at {daily_usage_pct:.1%} "
                f"(${daily_cost:.2f}/${self.config.daily_budget:.2f})"
            )
        elif daily_usage_pct >= self.config.budget_warning_threshold:
            status['warnings'].append(
                f"WARNING: Daily budget at {daily_usage_pct:.1%} "
                f"(${daily_cost:.2f}/${self.config.daily_budget:.2f})"
            )
        
        if monthly_usage_pct >= self.config.budget_critical_threshold:
            status['warnings'].append(
                f"CRITICAL: Monthly budget at {monthly_usage_pct:.1%} "
                f"(${monthly_cost:.2f}/${self.config.monthly_budget:.2f})"
            )
        elif monthly_usage_pct >= self.config.budget_warning_threshold:
            status['warnings'].append(
                f"WARNING: Monthly budget at {monthly_usage_pct:.1%} "
                f"(${monthly_cost:.2f}/${self.config.monthly_budget:.2f})"
            )
        
        # Log warnings
        for warning in status['warnings']:
            logger.warning(warning)
        
        return status
    
    def can_afford_job(
        self,
        estimated_cpu_hours: float = 0.0,
        estimated_gpu_hours: float = 0.0,
        estimated_memory_gb_hours: float = 0.0
    ) -> bool:
        """
        Check if job can be afforded within budget.
        
        Args:
            estimated_cpu_hours: Estimated CPU hours
            estimated_gpu_hours: Estimated GPU hours
            estimated_memory_gb_hours: Estimated memory GB-hours
        
        Returns:
            True if job can be afforded
        """
        # Estimate job cost
        estimated_cost = 0.0
        
        if estimated_cpu_hours > 0:
            estimated_cost += self.resource_costs[ResourceType.CPU_HOUR].calculate_cost(
                estimated_cpu_hours
            )
        
        if estimated_gpu_hours > 0:
            estimated_cost += self.resource_costs[ResourceType.GPU_HOUR].calculate_cost(
                estimated_gpu_hours
            )
        
        if estimated_memory_gb_hours > 0:
            estimated_cost += self.resource_costs[ResourceType.MEMORY_GB_HOUR].calculate_cost(
                estimated_memory_gb_hours
            )
        
        # Check against daily budget
        daily_cost = self.get_daily_cost()
        remaining_daily = self.config.daily_budget - daily_cost
        
        if estimated_cost > remaining_daily:
            logger.warning(
                f"Job cost ${estimated_cost:.2f} exceeds remaining daily budget "
                f"${remaining_daily:.2f}"
            )
            return False
        
        return True
    
    def get_cost_breakdown(self) -> Dict[ResourceType, float]:
        """Get cost breakdown by resource type."""
        breakdown = {}
        
        for resource_type in ResourceType:
            type_records = [
                record for record in self.usage_records
                if record.resource_type == resource_type
            ]
            breakdown[resource_type] = sum(record.cost for record in type_records)
        
        return breakdown
    
    def print_cost_summary(self) -> None:
        """Print cost summary."""
        daily_cost = self.get_daily_cost()
        monthly_cost = self.get_monthly_cost()
        breakdown = self.get_cost_breakdown()
        
        print("\n=== Cost Summary ===")
        print(f"Daily Cost: ${daily_cost:.2f} / ${self.config.daily_budget:.2f}")
        print(f"Monthly Cost: ${monthly_cost:.2f} / ${self.config.monthly_budget:.2f}")
        print("\nCost Breakdown:")
        for resource_type, cost in breakdown.items():
            if cost > 0:
                print(f"  {resource_type.value}: ${cost:.2f}")
        print("====================\n")
