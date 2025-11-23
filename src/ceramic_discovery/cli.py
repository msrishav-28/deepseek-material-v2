"""Command-line interface for Ceramic Armor Discovery Framework."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from ceramic_discovery.config import config


@click.group()
@click.version_option(version="0.1.0")
@click.option("--config-file", type=click.Path(exists=True), help="Custom configuration file")
def main(config_file: Optional[str]) -> None:
    """Ceramic Armor Discovery Framework CLI.
    
    A research-grade computational platform for high-throughput screening
    of ceramic armor materials using DFT and machine learning.
    """
    if config_file:
        click.echo(f"Loading configuration from: {config_file}")


@main.command()
@click.option("--check-deps", is_flag=True, help="Check all dependencies")
def init(check_deps: bool) -> None:
    """Initialize the research environment."""
    click.echo("Initializing Ceramic Armor Discovery Framework...")
    
    try:
        config.validate()
        click.echo("✓ Configuration validated")
        click.echo(f"✓ HDF5 data path: {config.hdf5.data_path}")
        click.echo(f"✓ Database URL: {config.database.url}")
        
        if check_deps:
            click.echo("\nChecking dependencies...")
            _check_dependencies()
        
        click.echo("\n✓ Environment initialized successfully")
    except ValueError as e:
        click.echo(f"✗ Configuration error: {e}", err=True)
        raise click.Abort()


@main.group()
def screen() -> None:
    """High-throughput screening commands."""
    pass


@screen.command("dopants")
@click.option("--base-system", "-b", required=True, 
              type=click.Choice(["SiC", "B4C", "WC", "TiC", "Al2O3"]),
              help="Base ceramic system")
@click.option("--dopants", "-d", required=True, help="Comma-separated dopant elements")
@click.option("--concentrations", "-c", default="0.01,0.02,0.03,0.05,0.10",
              help="Comma-separated concentrations (atomic fractions)")
@click.option("--output", "-o", type=click.Path(), default="./results/screening",
              help="Output directory")
@click.option("--parallel", "-p", type=int, default=None,
              help="Number of parallel jobs (default: from config)")
@click.option("--stability-threshold", type=float, default=0.1,
              help="Energy above hull threshold (eV/atom)")
def screen_dopants(base_system: str, dopants: str, concentrations: str,
                   output: str, parallel: Optional[int], stability_threshold: float) -> None:
    """Screen dopant candidates for a ceramic system.
    
    Example:
        ceramic-discovery screen dopants -b SiC -d "Ti,Zr,Hf" -c "0.01,0.05"
    """
    from ceramic_discovery.screening.screening_engine import ScreeningEngine
    
    dopant_list = [d.strip() for d in dopants.split(",")]
    conc_list = [float(c.strip()) for c in concentrations.split(",")]
    
    click.echo(f"Screening dopants for {base_system}")
    click.echo(f"Dopants: {', '.join(dopant_list)}")
    click.echo(f"Concentrations: {', '.join(map(str, conc_list))}")
    click.echo(f"Stability threshold: {stability_threshold} eV/atom")
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        engine = ScreeningEngine()
        results = engine.screen_dopants(
            base_system=base_system,
            dopants=dopant_list,
            concentrations=conc_list,
            stability_threshold=stability_threshold,
            max_parallel=parallel or config.computational.max_parallel_jobs
        )
        
        # Save results
        results_file = output_path / f"{base_system}_dopant_screening.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"\n✓ Screening complete. Results saved to: {results_file}")
        click.echo(f"✓ Viable candidates: {results.get('viable_count', 0)}")
        
    except Exception as e:
        click.echo(f"✗ Screening failed: {e}", err=True)
        raise click.Abort()


@screen.command("batch")
@click.option("--materials", "-m", type=click.Path(exists=True), required=True,
              help="Materials JSON file with screening configurations")
@click.option("--output", "-o", type=click.Path(), default="./results/batch",
              help="Output directory")
@click.option("--resume", is_flag=True, help="Resume from checkpoint")
def screen_batch(materials: str, output: str, resume: bool) -> None:
    """Run batch screening from configuration file.
    
    Example materials.json:
    {
      "base_systems": ["SiC", "B4C"],
      "dopants": ["Ti", "Zr"],
      "concentrations": [0.01, 0.05]
    }
    """
    from ceramic_discovery.screening.workflow_orchestrator import WorkflowOrchestrator
    
    click.echo(f"Loading batch configuration from: {materials}")
    
    with open(materials) as f:
        batch_config = json.load(f)
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        orchestrator = WorkflowOrchestrator()
        
        with click.progressbar(length=100, label="Screening progress") as bar:
            results = orchestrator.run_batch_screening(
                config=batch_config,
                output_dir=output_path,
                resume=resume,
                progress_callback=lambda p: bar.update(p)
            )
        
        click.echo(f"\n✓ Batch screening complete")
        click.echo(f"✓ Total materials screened: {results.get('total_screened', 0)}")
        click.echo(f"✓ Results saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Batch screening failed: {e}", err=True)
        raise click.Abort()


@main.group()
def ml() -> None:
    """Machine learning model commands."""
    pass


@ml.command("train")
@click.option("--data", "-d", type=click.Path(exists=True), required=True,
              help="Training data file (CSV or HDF5)")
@click.option("--target", "-t", default="v50", help="Target property to predict")
@click.option("--model-type", "-m", type=click.Choice(["rf", "xgboost", "svm", "ensemble"]),
              default="ensemble", help="Model type")
@click.option("--output", "-o", type=click.Path(), default="./results/models",
              help="Model output directory")
@click.option("--cv-folds", type=int, default=None, help="Cross-validation folds")
def ml_train(data: str, target: str, model_type: str, output: str, cv_folds: Optional[int]) -> None:
    """Train machine learning model for property prediction.
    
    Example:
        ceramic-discovery ml train -d data.csv -t v50 -m ensemble
    """
    from ceramic_discovery.ml.model_trainer import ModelTrainer
    
    click.echo(f"Training {model_type} model for {target} prediction")
    click.echo(f"Data: {data}")
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        trainer = ModelTrainer(
            model_type=model_type,
            cv_folds=cv_folds or config.ml.cross_validation_folds
        )
        
        results = trainer.train_from_file(
            data_file=data,
            target_property=target,
            output_dir=output_path
        )
        
        click.echo(f"\n✓ Training complete")
        click.echo(f"✓ R² score: {results['r2_score']:.3f}")
        click.echo(f"✓ RMSE: {results['rmse']:.3f}")
        click.echo(f"✓ Model saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Training failed: {e}", err=True)
        raise click.Abort()


@ml.command("predict")
@click.option("--model", "-m", type=click.Path(exists=True), required=True,
              help="Trained model file")
@click.option("--data", "-d", type=click.Path(exists=True), required=True,
              help="Input data file")
@click.option("--output", "-o", type=click.Path(), default="./results/predictions",
              help="Output directory")
@click.option("--uncertainty", is_flag=True, help="Include uncertainty quantification")
def ml_predict(model: str, data: str, output: str, uncertainty: bool) -> None:
    """Make predictions using trained model.
    
    Example:
        ceramic-discovery ml predict -m model.pkl -d test_data.csv --uncertainty
    """
    from ceramic_discovery.ballistics.ballistic_predictor import BallisticPredictor
    
    click.echo(f"Making predictions with model: {model}")
    click.echo(f"Input data: {data}")
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        predictor = BallisticPredictor.load_model(model)
        
        predictions = predictor.predict_from_file(
            data_file=data,
            include_uncertainty=uncertainty
        )
        
        # Save predictions
        output_file = output_path / "predictions.json"
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=2)
        
        click.echo(f"\n✓ Predictions complete")
        click.echo(f"✓ Results saved to: {output_file}")
        
    except Exception as e:
        click.echo(f"✗ Prediction failed: {e}", err=True)
        raise click.Abort()


@main.group()
def analyze() -> None:
    """Data analysis and visualization commands."""
    pass


@analyze.command("properties")
@click.option("--data", "-d", type=click.Path(exists=True), required=True,
              help="Materials data file")
@click.option("--output", "-o", type=click.Path(), default="./results/analysis",
              help="Output directory")
@click.option("--plot-types", default="all",
              help="Comma-separated plot types: correlation,distribution,pca,all")
def analyze_properties(data: str, output: str, plot_types: str) -> None:
    """Analyze material properties and generate visualizations.
    
    Example:
        ceramic-discovery analyze properties -d materials.csv --plot-types correlation,pca
    """
    from ceramic_discovery.analysis.property_analyzer import PropertyAnalyzer
    
    click.echo(f"Analyzing properties from: {data}")
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        analyzer = PropertyAnalyzer()
        
        plots = plot_types.split(",") if plot_types != "all" else ["all"]
        
        results = analyzer.analyze_from_file(
            data_file=data,
            output_dir=output_path,
            plot_types=plots
        )
        
        click.echo(f"\n✓ Analysis complete")
        click.echo(f"✓ Plots saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Analysis failed: {e}", err=True)
        raise click.Abort()


@analyze.command("mechanisms")
@click.option("--data", "-d", type=click.Path(exists=True), required=True,
              help="Materials data with ballistic performance")
@click.option("--output", "-o", type=click.Path(), default="./results/mechanisms",
              help="Output directory")
def analyze_mechanisms(data: str, output: str) -> None:
    """Analyze ballistic performance mechanisms.
    
    Example:
        ceramic-discovery analyze mechanisms -d results.csv
    """
    from ceramic_discovery.ballistics.mechanism_analyzer import MechanismAnalyzer
    
    click.echo(f"Analyzing mechanisms from: {data}")
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        analyzer = MechanismAnalyzer()
        
        results = analyzer.analyze_from_file(
            data_file=data,
            output_dir=output_path
        )
        
        click.echo(f"\n✓ Mechanism analysis complete")
        click.echo(f"✓ Key drivers identified: {len(results.get('key_drivers', []))}")
        click.echo(f"✓ Results saved to: {output_path}")
        
    except Exception as e:
        click.echo(f"✗ Analysis failed: {e}", err=True)
        raise click.Abort()


@main.group()
def export() -> None:
    """Data export and reporting commands."""
    pass


@export.command("results")
@click.option("--data", "-d", type=click.Path(exists=True), required=True,
              help="Results data file")
@click.option("--format", "-f", type=click.Choice(["json", "csv", "hdf5", "excel"]),
              default="json", help="Export format")
@click.option("--output", "-o", type=click.Path(), required=True,
              help="Output file path")
@click.option("--include-uncertainty", is_flag=True, help="Include uncertainty data")
def export_results(data: str, format: str, output: str, include_uncertainty: bool) -> None:
    """Export results in various formats.
    
    Example:
        ceramic-discovery export results -d screening.json -f csv -o results.csv
    """
    from ceramic_discovery.reporting.data_exporter import DataExporter
    
    click.echo(f"Exporting results to {format} format")
    
    try:
        exporter = DataExporter()
        
        exporter.export(
            input_file=data,
            output_file=output,
            format=format,
            include_uncertainty=include_uncertainty
        )
        
        click.echo(f"✓ Export complete: {output}")
        
    except Exception as e:
        click.echo(f"✗ Export failed: {e}", err=True)
        raise click.Abort()


@export.command("report")
@click.option("--data", "-d", type=click.Path(exists=True), required=True,
              help="Results data file")
@click.option("--template", "-t", type=click.Choice(["screening", "ml", "full"]),
              default="full", help="Report template")
@click.option("--output", "-o", type=click.Path(), required=True,
              help="Output report file (PDF or HTML)")
def export_report(data: str, template: str, output: str) -> None:
    """Generate publication-quality research report.
    
    Example:
        ceramic-discovery export report -d results.json -t full -o report.pdf
    """
    from ceramic_discovery.reporting.report_generator import ReportGenerator
    
    click.echo(f"Generating {template} report")
    
    try:
        generator = ReportGenerator()
        
        generator.generate_report(
            data_file=data,
            template=template,
            output_file=output
        )
        
        click.echo(f"✓ Report generated: {output}")
        
    except Exception as e:
        click.echo(f"✗ Report generation failed: {e}", err=True)
        raise click.Abort()


@main.group()
def monitor() -> None:
    """Monitoring and observability commands."""
    pass


@monitor.command("performance")
@click.option("--summary", is_flag=True, help="Show performance summary")
@click.option("--system", is_flag=True, help="Show system resource info")
def monitor_performance(summary: bool, system: bool) -> None:
    """Monitor computational performance.
    
    Example:
        ceramic-discovery monitor performance --summary
        ceramic-discovery monitor performance --system
    """
    from ceramic_discovery.monitoring.performance_monitor import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    
    if system:
        click.echo("System Resource Information:")
        click.echo("=" * 60)
        info = monitor.get_system_info()
        
        click.echo(f"\nCPU:")
        click.echo(f"  Cores: {info['cpu']['count']}")
        click.echo(f"  Usage: {info['cpu']['percent']:.1f}%")
        
        click.echo(f"\nMemory:")
        click.echo(f"  Total: {info['memory']['total_gb']:.1f} GB")
        click.echo(f"  Available: {info['memory']['available_gb']:.1f} GB")
        click.echo(f"  Usage: {info['memory']['percent']:.1f}%")
        
        click.echo(f"\nDisk:")
        click.echo(f"  Total: {info['disk']['total_gb']:.1f} GB")
        click.echo(f"  Free: {info['disk']['free_gb']:.1f} GB")
        click.echo(f"  Usage: {info['disk']['percent']:.1f}%")
    
    if summary:
        click.echo("\nPerformance Summary:")
        click.echo("=" * 60)
        summary_data = monitor.get_summary()
        
        if "message" in summary_data:
            click.echo(summary_data["message"])
        else:
            click.echo(f"Total Operations: {summary_data['total_operations']}")
            click.echo(f"Total Duration: {summary_data['total_duration_seconds']:.1f}s")
            click.echo(f"Average CPU: {summary_data['average_cpu_percent']:.1f}%")
            click.echo(f"Average Memory: {summary_data['average_memory_mb']:.1f} MB")
            click.echo(f"Peak Memory: {summary_data['peak_memory_mb']:.1f} MB")


@monitor.command("progress")
@click.option("--workflow-id", "-w", help="Show specific workflow progress")
@click.option("--active", is_flag=True, help="Show all active workflows")
@click.option("--summary", is_flag=True, help="Show progress summary")
def monitor_progress(workflow_id: Optional[str], active: bool, summary: bool) -> None:
    """Monitor research progress.
    
    Example:
        ceramic-discovery monitor progress --active
        ceramic-discovery monitor progress --workflow-id abc123
        ceramic-discovery monitor progress --summary
    """
    from ceramic_discovery.monitoring.progress_tracker import ProgressTracker
    
    tracker = ProgressTracker()
    
    if workflow_id:
        progress = tracker.get_progress(workflow_id)
        if progress:
            click.echo(f"Workflow: {progress.workflow_id}")
            click.echo(f"Type: {progress.workflow_type}")
            click.echo(f"Status: {progress.status}")
            click.echo(f"Progress: {progress.progress_percent:.1f}%")
            click.echo(f"Completed: {progress.completed_items}/{progress.total_items}")
            click.echo(f"Failed: {progress.failed_items}")
            click.echo(f"Viable Candidates: {progress.viable_candidates}")
        else:
            click.echo(f"Workflow {workflow_id} not found")
    
    if active:
        workflows = tracker.get_all_active()
        if workflows:
            click.echo("Active Workflows:")
            click.echo("=" * 60)
            for wf in workflows:
                click.echo(f"\n{wf.workflow_id} ({wf.workflow_type})")
                click.echo(f"  Progress: {wf.progress_percent:.1f}%")
                click.echo(f"  Status: {wf.status}")
        else:
            click.echo("No active workflows")
    
    if summary:
        summary_data = tracker.get_summary()
        if "message" in summary_data:
            click.echo(summary_data["message"])
        else:
            click.echo("Progress Summary:")
            click.echo("=" * 60)
            click.echo(f"Total Workflows: {summary_data['total_workflows']}")
            click.echo(f"Active: {summary_data['active_workflows']}")
            click.echo(f"Completed: {summary_data['completed_workflows']}")
            click.echo(f"Items Processed: {summary_data['total_items_processed']}")
            click.echo(f"Viable Candidates: {summary_data['total_viable_candidates']}")


@monitor.command("data-quality")
@click.option("--dataset", "-d", required=True, help="Dataset to check")
@click.option("--summary", is_flag=True, help="Show quality summary")
def monitor_data_quality(dataset: str, summary: bool) -> None:
    """Monitor data quality.
    
    Example:
        ceramic-discovery monitor data-quality --dataset materials.csv --summary
    """
    from ceramic_discovery.monitoring.data_quality_monitor import DataQualityMonitor
    import pandas as pd
    
    monitor = DataQualityMonitor()
    
    # Load dataset
    try:
        if dataset.endswith('.csv'):
            df = pd.read_csv(dataset)
            data = df.to_dict('list')
        else:
            click.echo(f"Unsupported format: {dataset}")
            return
        
        # Check quality
        metrics = monitor.check_data_quality(data, dataset)
        
        click.echo(f"Data Quality Report: {dataset}")
        click.echo("=" * 60)
        click.echo(f"Total Records: {metrics.total_records}")
        click.echo(f"Complete Records: {metrics.complete_records}")
        click.echo(f"Missing Values: {metrics.missing_values_percent:.1f}%")
        click.echo(f"Invalid Values: {metrics.invalid_values_count}")
        click.echo(f"Out of Range: {metrics.out_of_range_count}")
        click.echo(f"Outliers: {metrics.outliers_percent:.1f}%")
        
        if metrics.issues:
            click.echo("\nIssues:")
            for issue in metrics.issues:
                click.secho(f"  ✗ {issue}", fg="red")
        
        if metrics.warnings:
            click.echo("\nWarnings:")
            for warning in metrics.warnings:
                click.secho(f"  ⚠ {warning}", fg="yellow")
        
        if summary:
            summary_data = monitor.get_summary()
            if "message" not in summary_data:
                click.echo("\nQuality Trends:")
                click.echo(f"Total Checks: {summary_data['total_checks']}")
    
    except Exception as e:
        click.echo(f"Error checking data quality: {e}", err=True)


@monitor.command("alerts")
@click.option("--active", is_flag=True, help="Show active alerts")
@click.option("--severity", type=click.Choice(["info", "warning", "error", "critical"]),
              help="Filter by severity")
@click.option("--summary", is_flag=True, help="Show alert summary")
def monitor_alerts(active: bool, severity: Optional[str], summary: bool) -> None:
    """Monitor system alerts.
    
    Example:
        ceramic-discovery monitor alerts --active
        ceramic-discovery monitor alerts --severity error
        ceramic-discovery monitor alerts --summary
    """
    from ceramic_discovery.monitoring.alerting import AlertManager, AlertSeverity
    
    manager = AlertManager()
    
    if active:
        severity_filter = AlertSeverity(severity) if severity else None
        alerts = manager.get_active_alerts(severity_filter)
        
        if alerts:
            click.echo("Active Alerts:")
            click.echo("=" * 60)
            for alert in alerts:
                color = {
                    AlertSeverity.INFO: "blue",
                    AlertSeverity.WARNING: "yellow",
                    AlertSeverity.ERROR: "red",
                    AlertSeverity.CRITICAL: "red"
                }[alert.severity]
                
                click.secho(f"\n[{alert.severity.value.upper()}] {alert.title}", fg=color, bold=True)
                click.echo(f"  {alert.message}")
                click.echo(f"  Source: {alert.source}")
                click.echo(f"  Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            click.echo("No active alerts")
    
    if summary:
        summary_data = manager.get_summary()
        click.echo("Alert Summary:")
        click.echo("=" * 60)
        click.echo(f"Active Alerts: {summary_data['active_alerts']}")
        click.echo(f"Total Alerts: {summary_data['total_alerts']}")
        click.echo(f"Resolved: {summary_data['resolved_alerts']}")
        
        click.echo("\nBy Severity:")
        for severity, count in summary_data['by_severity'].items():
            if count > 0:
                click.echo(f"  {severity}: {count}")


@main.command()
@click.option("--verbose", "-v", is_flag=True, help="Show detailed status")
def status(verbose: bool) -> None:
    """Show system status and configuration."""
    click.echo("Ceramic Armor Discovery Framework Status")
    click.echo("=" * 60)
    
    # Configuration
    click.echo("\n[Configuration]")
    click.echo(f"Database: {config.database.url}")
    click.echo(f"Redis: {config.redis.url}")
    click.echo(f"HDF5 Path: {config.hdf5.data_path}")
    
    # Computational settings
    click.echo("\n[Computational Settings]")
    click.echo(f"Max Parallel Jobs: {config.computational.max_parallel_jobs}")
    click.echo(f"Random Seed: {config.ml.random_seed}")
    click.echo(f"CV Folds: {config.ml.cross_validation_folds}")
    
    # ML targets
    click.echo("\n[ML Performance Targets]")
    click.echo(f"Target R² Range: {config.ml.target_r2_min:.2f} - {config.ml.target_r2_max:.2f}")
    click.echo(f"Bootstrap Samples: {config.ml.bootstrap_samples}")
    
    if verbose:
        click.echo("\n[HPC Configuration]")
        click.echo(f"Scheduler: {config.hpc.scheduler}")
        click.echo(f"Partition: {config.hpc.partition}")
        click.echo(f"Max Nodes: {config.hpc.max_nodes}")
        
        click.echo("\n[Materials Project]")
        api_key_status = "✓ Configured" if config.materials_project.api_key else "✗ Not configured"
        click.echo(f"API Key: {api_key_status}")
        click.echo(f"Rate Limit: {config.materials_project.rate_limit_per_second} req/s")
    
    click.echo("\n" + "=" * 60)


@main.command()
@click.option("--check-type", "-t", 
              type=click.Choice(["full", "liveness", "readiness"]),
              default="full",
              help="Type of health check to perform")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def health(check_type: str, output_json: bool) -> None:
    """Perform health check for deployment monitoring.
    
    Examples:
        ceramic-discovery health --check-type liveness
        ceramic-discovery health --check-type readiness --json
        ceramic-discovery health --check-type full
    """
    from ceramic_discovery.health import HealthChecker
    
    checker = HealthChecker()
    
    if check_type == "liveness":
        result = checker.liveness_check()
    elif check_type == "readiness":
        result = checker.readiness_check()
    else:
        result = checker.full_health_check()
    
    if output_json:
        click.echo(json.dumps(result, indent=2))
    else:
        _print_health_result(result, check_type)
    
    # Exit with appropriate code
    status = result.get("status", "unknown")
    if status in ["healthy", "ready"]:
        sys.exit(0)
    elif status == "degraded":
        sys.exit(0)  # Still operational
    else:
        sys.exit(1)  # Unhealthy


@main.command()
@click.option("--scenario", "-s", required=True,
              type=click.Choice(["development", "production", "hpc"]),
              help="Configuration scenario")
@click.option("--output", "-o", type=click.Path(), default=".env.custom",
              help="Output configuration file")
def configure(scenario: str, output: str) -> None:
    """Generate configuration for different research scenarios.
    
    Example:
        ceramic-discovery configure -s hpc -o .env.hpc
    """
    click.echo(f"Generating {scenario} configuration")
    
    configs = {
        "development": {
            "DATABASE_URL": "postgresql://localhost:5432/ceramic_dev",
            "REDIS_URL": "redis://localhost:6379/0",
            "MAX_PARALLEL_JOBS": "2",
            "LOG_LEVEL": "DEBUG"
        },
        "production": {
            "DATABASE_URL": "postgresql://prod-server:5432/ceramic_prod",
            "REDIS_URL": "redis://prod-server:6379/0",
            "MAX_PARALLEL_JOBS": "8",
            "LOG_LEVEL": "INFO"
        },
        "hpc": {
            "HPC_SCHEDULER": "slurm",
            "HPC_PARTITION": "compute",
            "HPC_MAX_NODES": "16",
            "MAX_PARALLEL_JOBS": "64",
            "LOG_LEVEL": "INFO"
        }
    }
    
    with open(output, "w") as f:
        f.write(f"# Configuration for {scenario} scenario\n\n")
        for key, value in configs[scenario].items():
            f.write(f"{key}={value}\n")
    
    click.echo(f"✓ Configuration saved to: {output}")
    click.echo(f"  Load with: export $(cat {output} | xargs)")


def _check_dependencies() -> None:
    """Check if all required dependencies are available."""
    deps = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "pymatgen": "pymatgen",
        "click": "click"
    }
    
    for name, module in deps.items():
        try:
            __import__(module)
            click.echo(f"  ✓ {name}")
        except ImportError:
            click.echo(f"  ✗ {name} (missing)", err=True)


def _print_health_result(result: dict, check_type: str) -> None:
    """Print health check result in human-readable format."""
    status = result.get("status", "unknown")
    
    # Status indicator
    if status in ["healthy", "ready"]:
        indicator = "✓"
        color = "green"
    elif status == "degraded":
        indicator = "⚠"
        color = "yellow"
    else:
        indicator = "✗"
        color = "red"
    
    click.secho(f"\n{indicator} Status: {status.upper()}", fg=color, bold=True)
    
    if check_type == "full":
        click.echo("\nComponent Status:")
        click.echo("-" * 60)
        
        for component, details in result.get("checks", {}).items():
            comp_status = details.get("status", "unknown")
            if comp_status == "healthy":
                click.secho(f"  ✓ {component.title()}: {comp_status}", fg="green")
            elif comp_status == "degraded":
                click.secho(f"  ⚠ {component.title()}: {comp_status}", fg="yellow")
            else:
                click.secho(f"  ✗ {component.title()}: {comp_status}", fg="red")
                if "error" in details:
                    click.echo(f"    Error: {details['error']}")
        
        # System info
        system = result.get("system", {})
        click.echo("\nSystem Information:")
        click.echo("-" * 60)
        click.echo(f"  Uptime: {system.get('uptime_seconds', 0):.0f} seconds")
        click.echo(f"  Timestamp: {system.get('timestamp', 'unknown')}")
    
    elif check_type in ["liveness", "readiness"]:
        click.echo(f"Message: {result.get('message', 'No message')}")
        click.echo(f"Timestamp: {result.get('timestamp', 'unknown')}")


if __name__ == "__main__":
    main()
