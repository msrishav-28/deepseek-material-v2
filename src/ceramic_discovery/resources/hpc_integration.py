"""HPC cluster integration with SLURM and PBS job schedulers."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
import logging
import subprocess
import re

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of HPC job."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class SlurmConfig:
    """Configuration for SLURM scheduler."""
    
    partition: str = "compute"
    account: Optional[str] = None
    qos: Optional[str] = None
    time_limit: str = "24:00:00"
    nodes: int = 1
    ntasks_per_node: int = 32
    cpus_per_task: int = 1
    mem_per_cpu: str = "4G"
    output_dir: Path = Path("./logs/slurm")
    
    def __post_init__(self):
        """Create output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PBSConfig:
    """Configuration for PBS scheduler."""
    
    queue: str = "batch"
    walltime: str = "24:00:00"
    nodes: int = 1
    ppn: int = 32
    mem: str = "128gb"
    output_dir: Path = Path("./logs/pbs")
    
    def __post_init__(self):
        """Create output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class HPCJob:
    """HPC job information."""
    
    job_id: str
    job_name: str
    status: JobStatus
    submit_time: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    nodes: int = 1
    cpus: int = 1
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HPCScheduler:
    """
    HPC cluster integration with SLURM and PBS job schedulers.
    
    Provides unified interface for submitting and monitoring jobs
    on HPC clusters using SLURM or PBS.
    """
    
    def __init__(
        self,
        scheduler_type: str = "slurm",
        slurm_config: Optional[SlurmConfig] = None,
        pbs_config: Optional[PBSConfig] = None
    ):
        """
        Initialize HPC scheduler.
        
        Args:
            scheduler_type: Type of scheduler ('slurm' or 'pbs')
            slurm_config: SLURM configuration
            pbs_config: PBS configuration
        """
        self.scheduler_type = scheduler_type.lower()
        
        if self.scheduler_type not in ['slurm', 'pbs']:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        self.slurm_config = slurm_config or SlurmConfig()
        self.pbs_config = pbs_config or PBSConfig()
        
        self.jobs: Dict[str, HPCJob] = {}
        
        logger.info(f"Initialized HPCScheduler with {self.scheduler_type}")
    
    def submit_slurm_job(
        self,
        job_name: str,
        command: str,
        script_path: Optional[Path] = None,
        custom_options: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Submit job to SLURM scheduler.
        
        Args:
            job_name: Name of job
            command: Command to execute
            script_path: Optional path to save script
            custom_options: Optional custom SLURM options
        
        Returns:
            Job ID
        """
        # Create SLURM script
        script_content = self._create_slurm_script(
            job_name, command, custom_options
        )
        
        # Save script
        if script_path is None:
            script_path = self.slurm_config.output_dir / f"{job_name}.sh"
        
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Submit job
        try:
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse job ID from output
            # Expected format: "Submitted batch job 12345"
            match = re.search(r'Submitted batch job (\d+)', result.stdout)
            if match:
                job_id = match.group(1)
            else:
                raise ValueError(f"Could not parse job ID from: {result.stdout}")
            
            # Record job
            job = HPCJob(
                job_id=job_id,
                job_name=job_name,
                status=JobStatus.PENDING,
                submit_time=datetime.now(),
                nodes=self.slurm_config.nodes,
                cpus=self.slurm_config.ntasks_per_node * self.slurm_config.cpus_per_task,
                metadata={'script_path': str(script_path)}
            )
            
            self.jobs[job_id] = job
            
            logger.info(f"Submitted SLURM job {job_id}: {job_name}")
            
            return job_id
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit SLURM job: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("SLURM not available. Is sbatch in PATH?")
            raise
    
    def submit_pbs_job(
        self,
        job_name: str,
        command: str,
        script_path: Optional[Path] = None,
        custom_options: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Submit job to PBS scheduler.
        
        Args:
            job_name: Name of job
            command: Command to execute
            script_path: Optional path to save script
            custom_options: Optional custom PBS options
        
        Returns:
            Job ID
        """
        # Create PBS script
        script_content = self._create_pbs_script(
            job_name, command, custom_options
        )
        
        # Save script
        if script_path is None:
            script_path = self.pbs_config.output_dir / f"{job_name}.sh"
        
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Submit job
        try:
            result = subprocess.run(
                ['qsub', str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse job ID from output
            job_id = result.stdout.strip()
            
            # Record job
            job = HPCJob(
                job_id=job_id,
                job_name=job_name,
                status=JobStatus.PENDING,
                submit_time=datetime.now(),
                nodes=self.pbs_config.nodes,
                cpus=self.pbs_config.ppn,
                metadata={'script_path': str(script_path)}
            )
            
            self.jobs[job_id] = job
            
            logger.info(f"Submitted PBS job {job_id}: {job_name}")
            
            return job_id
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit PBS job: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("PBS not available. Is qsub in PATH?")
            raise
    
    def submit_job(
        self,
        job_name: str,
        command: str,
        script_path: Optional[Path] = None,
        custom_options: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Submit job using configured scheduler.
        
        Args:
            job_name: Name of job
            command: Command to execute
            script_path: Optional path to save script
            custom_options: Optional custom options
        
        Returns:
            Job ID
        """
        if self.scheduler_type == 'slurm':
            return self.submit_slurm_job(job_name, command, script_path, custom_options)
        elif self.scheduler_type == 'pbs':
            return self.submit_pbs_job(job_name, command, script_path, custom_options)
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")
    
    def _create_slurm_script(
        self,
        job_name: str,
        command: str,
        custom_options: Optional[Dict[str, str]] = None
    ) -> str:
        """Create SLURM batch script."""
        config = self.slurm_config
        
        script_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={config.partition}",
            f"#SBATCH --time={config.time_limit}",
            f"#SBATCH --nodes={config.nodes}",
            f"#SBATCH --ntasks-per-node={config.ntasks_per_node}",
            f"#SBATCH --cpus-per-task={config.cpus_per_task}",
            f"#SBATCH --mem-per-cpu={config.mem_per_cpu}",
            f"#SBATCH --output={config.output_dir}/{job_name}_%j.out",
            f"#SBATCH --error={config.output_dir}/{job_name}_%j.err",
        ]
        
        if config.account:
            script_lines.append(f"#SBATCH --account={config.account}")
        
        if config.qos:
            script_lines.append(f"#SBATCH --qos={config.qos}")
        
        # Add custom options
        if custom_options:
            for key, value in custom_options.items():
                script_lines.append(f"#SBATCH --{key}={value}")
        
        script_lines.extend([
            "",
            "# Load modules or activate environment here",
            "# module load python/3.11",
            "# source activate ceramic-discovery",
            "",
            "# Execute command",
            command,
        ])
        
        return "\n".join(script_lines)
    
    def _create_pbs_script(
        self,
        job_name: str,
        command: str,
        custom_options: Optional[Dict[str, str]] = None
    ) -> str:
        """Create PBS batch script."""
        config = self.pbs_config
        
        script_lines = [
            "#!/bin/bash",
            f"#PBS -N {job_name}",
            f"#PBS -q {config.queue}",
            f"#PBS -l walltime={config.walltime}",
            f"#PBS -l nodes={config.nodes}:ppn={config.ppn}",
            f"#PBS -l mem={config.mem}",
            f"#PBS -o {config.output_dir}/{job_name}.out",
            f"#PBS -e {config.output_dir}/{job_name}.err",
        ]
        
        # Add custom options
        if custom_options:
            for key, value in custom_options.items():
                script_lines.append(f"#PBS -{key} {value}")
        
        script_lines.extend([
            "",
            "# Change to working directory",
            "cd $PBS_O_WORKDIR",
            "",
            "# Load modules or activate environment here",
            "# module load python/3.11",
            "# source activate ceramic-discovery",
            "",
            "# Execute command",
            command,
        ])
        
        return "\n".join(script_lines)
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """
        Get status of job.
        
        Args:
            job_id: Job ID
        
        Returns:
            Job status
        """
        if self.scheduler_type == 'slurm':
            return self._get_slurm_job_status(job_id)
        elif self.scheduler_type == 'pbs':
            return self._get_pbs_job_status(job_id)
        else:
            return JobStatus.UNKNOWN
    
    def _get_slurm_job_status(self, job_id: str) -> JobStatus:
        """Get SLURM job status."""
        try:
            result = subprocess.run(
                ['squeue', '-j', job_id, '-h', '-o', '%T'],
                capture_output=True,
                text=True,
                check=True
            )
            
            status_str = result.stdout.strip().upper()
            
            status_map = {
                'PENDING': JobStatus.PENDING,
                'RUNNING': JobStatus.RUNNING,
                'COMPLETED': JobStatus.COMPLETED,
                'FAILED': JobStatus.FAILED,
                'CANCELLED': JobStatus.CANCELLED,
            }
            
            return status_map.get(status_str, JobStatus.UNKNOWN)
        
        except subprocess.CalledProcessError:
            # Job not in queue, check if completed
            try:
                result = subprocess.run(
                    ['sacct', '-j', job_id, '-n', '-o', 'State'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                status_str = result.stdout.strip().upper()
                
                if 'COMPLETED' in status_str:
                    return JobStatus.COMPLETED
                elif 'FAILED' in status_str:
                    return JobStatus.FAILED
                elif 'CANCELLED' in status_str:
                    return JobStatus.CANCELLED
            
            except subprocess.CalledProcessError:
                pass
            
            return JobStatus.UNKNOWN
    
    def _get_pbs_job_status(self, job_id: str) -> JobStatus:
        """Get PBS job status."""
        try:
            result = subprocess.run(
                ['qstat', job_id],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse status from qstat output
            lines = result.stdout.strip().split('\n')
            if len(lines) > 2:
                status_char = lines[2].split()[-2]
                
                status_map = {
                    'Q': JobStatus.PENDING,
                    'R': JobStatus.RUNNING,
                    'C': JobStatus.COMPLETED,
                    'E': JobStatus.FAILED,
                }
                
                return status_map.get(status_char, JobStatus.UNKNOWN)
        
        except subprocess.CalledProcessError:
            return JobStatus.UNKNOWN
        
        return JobStatus.UNKNOWN
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel job.
        
        Args:
            job_id: Job ID
        
        Returns:
            True if successful
        """
        try:
            if self.scheduler_type == 'slurm':
                subprocess.run(['scancel', job_id], check=True)
            elif self.scheduler_type == 'pbs':
                subprocess.run(['qdel', job_id], check=True)
            
            # Update job status
            if job_id in self.jobs:
                self.jobs[job_id].status = JobStatus.CANCELLED
            
            logger.info(f"Cancelled job {job_id}")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def get_all_jobs(self) -> List[HPCJob]:
        """Get all tracked jobs."""
        return list(self.jobs.values())
    
    def update_job_statuses(self) -> None:
        """Update status of all tracked jobs."""
        for job_id in self.jobs:
            status = self.get_job_status(job_id)
            self.jobs[job_id].status = status
