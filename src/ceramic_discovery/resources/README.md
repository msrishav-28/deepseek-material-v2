# Resource Management Module

This module provides resource management capabilities for computational resources, cost tracking, and HPC integration.

## Components

### CostManager
Cost-aware job scheduling for external APIs and computational resources.

**Features:**
- API rate limiting
- Cost tracking by resource type
- Budget enforcement (daily/monthly)
- Alert thresholds
- Cost breakdown and reporting

**Usage:**
```python
from ceramic_discovery.resources import CostManager, CostConfig

config = CostConfig(
    daily_budget=100.0,
    monthly_budget=2000.0,
    materials_project_rate_limit_per_hour=1000
)

cost_manager = CostManager(config)

# Check if API call can be made
if cost_manager.can_make_api_call():
    # Make API call
    cost_manager.record_api_call(job_id="job_1")

# Record compute usage
cost_manager.record_compute_usage(
    cpu_hours=5.0,
    gpu_hours=2.0,
    memory_gb_hours=10.0
)

# Check budget status
status = cost_manager.check_budget_status()
```

### HPCScheduler
HPC cluster integration with SLURM and PBS job schedulers.

**Features:**
- Unified interface for SLURM and PBS
- Automatic script generation
- Job submission and monitoring
- Status tracking
- Job cancellation

**Usage:**
```python
from ceramic_discovery.resources import HPCScheduler, SlurmConfig

config = SlurmConfig(
    partition="compute",
    time_limit="24:00:00",
    nodes=4,
    ntasks_per_node=32
)

scheduler = HPCScheduler(scheduler_type="slurm", slurm_config=config)

# Submit job
job_id = scheduler.submit_job(
    job_name="dft_screening",
    command="python run_screening.py"
)

# Check status
status = scheduler.get_job_status(job_id)

# Cancel job
scheduler.cancel_job(job_id)
```

### ResourceMonitor
Resource usage monitoring and optimization.

**Features:**
- CPU, memory, disk, and network monitoring
- Real-time usage tracking
- Function execution monitoring
- System information
- Usage statistics and recommendations

**Usage:**
```python
from ceramic_discovery.resources import ResourceMonitor

monitor = ResourceMonitor()

# Get current usage
usage = monitor.get_current_usage()

# Monitor function execution
result, stats = monitor.monitor_function(my_function, *args)

# Check resource limits
warnings = monitor.check_resource_limits(
    cpu_threshold=90.0,
    memory_threshold=90.0
)

# Get optimization recommendations
recommendations = monitor.optimize_recommendations()
```

### BudgetTracker
Computational budget tracking and management.

**Features:**
- Multiple budget support
- Expense tracking by category
- Alert thresholds
- Budget forecasting
- Daily burn rate calculation
- State persistence

**Usage:**
```python
from ceramic_discovery.resources import BudgetTracker

tracker = BudgetTracker()

# Create budget
tracker.create_budget(
    name="research_project",
    total_amount=1000.0,
    duration_days=30
)

# Record expense
tracker.record_expense(
    budget_name="research_project",
    amount=250.0,
    description="GPU compute time",
    category="compute"
)

# Get budget status
status = tracker.get_budget_status("research_project")

# Save state
tracker.save_state()
```

## Resource Types

The system tracks the following resource types:

- **API_CALL**: External API calls (e.g., Materials Project)
- **CPU_HOUR**: CPU compute hours
- **GPU_HOUR**: GPU compute hours
- **MEMORY_GB_HOUR**: Memory usage in GB-hours
- **STORAGE_GB**: Storage in GB
- **NETWORK_GB**: Network transfer in GB

## Cost Configuration

Default cost values (can be customized):

```python
CostConfig(
    # API costs
    materials_project_cost_per_call=0.0,  # Free but rate-limited
    materials_project_rate_limit_per_hour=1000,
    
    # Compute costs
    cpu_cost_per_hour=0.10,
    gpu_cost_per_hour=1.00,
    memory_cost_per_gb_hour=0.01,
    storage_cost_per_gb=0.02,
    
    # Budget limits
    daily_budget=100.0,
    monthly_budget=2000.0,
    
    # Alert thresholds
    budget_warning_threshold=0.8,  # 80%
    budget_critical_threshold=0.95  # 95%
)
```

## HPC Integration

### SLURM Configuration

```python
SlurmConfig(
    partition="compute",
    account="my_account",
    qos="normal",
    time_limit="24:00:00",
    nodes=1,
    ntasks_per_node=32,
    cpus_per_task=1,
    mem_per_cpu="4G"
)
```

### PBS Configuration

```python
PBSConfig(
    queue="batch",
    walltime="24:00:00",
    nodes=1,
    ppn=32,
    mem="128gb"
)
```

## Best Practices

1. **Set realistic budgets** based on your computational needs
2. **Monitor resource usage** regularly to identify inefficiencies
3. **Use cost-aware scheduling** to prevent unexpected expenses
4. **Track expenses by category** for better budget allocation
5. **Enable alerts** to catch budget overruns early
6. **Save budget state** regularly for persistence
7. **Use HPC integration** for large-scale computations

## Requirements

- `psutil` (for resource monitoring)
- SLURM or PBS (for HPC integration, optional)

## Examples

See `examples/performance_optimization_example.py` for comprehensive usage examples.
