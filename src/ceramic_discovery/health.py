"""Health check endpoints for containerized deployment."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import click


class HealthChecker:
    """Health check system for deployment monitoring."""
    
    def __init__(self):
        """Initialize health checker."""
        self.start_time = datetime.now()
    
    def check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            from ceramic_discovery.config import config
            from ceramic_discovery.dft.database_manager import DatabaseManager
            
            db = DatabaseManager()
            # Simple connectivity check
            db.engine.connect()
            
            return {
                "status": "healthy",
                "url": config.database.url.split("@")[-1] if "@" in config.database.url else "configured",
                "message": "Database connection successful"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Database connection failed"
            }
    
    def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        try:
            from ceramic_discovery.config import config
            import redis
            
            r = redis.from_url(config.redis.url)
            r.ping()
            
            return {
                "status": "healthy",
                "url": config.redis.url,
                "message": "Redis connection successful"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Redis connection failed"
            }
    
    def check_storage(self) -> Dict[str, Any]:
        """Check storage paths."""
        try:
            from ceramic_discovery.config import config
            
            paths = {
                "hdf5": Path(config.hdf5.data_path),
                "results": Path("./results"),
                "logs": Path("./logs")
            }
            
            status = {}
            all_ok = True
            
            for name, path in paths.items():
                if path.exists() and path.is_dir():
                    status[name] = {
                        "status": "healthy",
                        "path": str(path),
                        "writable": path.stat().st_mode & 0o200 != 0
                    }
                else:
                    status[name] = {
                        "status": "unhealthy",
                        "path": str(path),
                        "error": "Path does not exist or is not a directory"
                    }
                    all_ok = False
            
            return {
                "status": "healthy" if all_ok else "degraded",
                "paths": status,
                "message": "Storage check complete"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Storage check failed"
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies."""
        deps = {
            "numpy": "numpy",
            "pandas": "pandas",
            "scikit-learn": "sklearn",
            "pymatgen": "pymatgen",
            "ase": "ase",
            "xgboost": "xgboost",
            "sqlalchemy": "sqlalchemy",
            "redis": "redis"
        }
        
        status = {}
        all_ok = True
        
        for name, module in deps.items():
            try:
                mod = __import__(module)
                version = getattr(mod, "__version__", "unknown")
                status[name] = {
                    "status": "healthy",
                    "version": version
                }
            except ImportError as e:
                status[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                all_ok = False
        
        return {
            "status": "healthy" if all_ok else "unhealthy",
            "dependencies": status,
            "message": "Dependency check complete"
        }
    
    def check_api_keys(self) -> Dict[str, Any]:
        """Check API key configuration."""
        try:
            from ceramic_discovery.config import config
            
            keys = {
                "materials_project": bool(config.materials_project.api_key)
            }
            
            return {
                "status": "healthy" if all(keys.values()) else "degraded",
                "configured": keys,
                "message": "API key check complete"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "API key check failed"
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "uptime_seconds": uptime,
            "timestamp": datetime.now().isoformat()
        }
    
    def full_health_check(self) -> Dict[str, Any]:
        """Perform full health check."""
        checks = {
            "database": self.check_database(),
            "redis": self.check_redis(),
            "storage": self.check_storage(),
            "dependencies": self.check_dependencies(),
            "api_keys": self.check_api_keys()
        }
        
        # Determine overall status
        statuses = [check["status"] for check in checks.values()]
        if all(s == "healthy" for s in statuses):
            overall_status = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "checks": checks,
            "system": self.get_system_info()
        }
    
    def liveness_check(self) -> Dict[str, Any]:
        """Simple liveness check (is the application running?)."""
        return {
            "status": "healthy",
            "message": "Application is running",
            "timestamp": datetime.now().isoformat()
        }
    
    def readiness_check(self) -> Dict[str, Any]:
        """Readiness check (is the application ready to serve requests?)."""
        # Check critical components only
        db_status = self.check_database()
        deps_status = self.check_dependencies()
        
        if db_status["status"] == "healthy" and deps_status["status"] == "healthy":
            return {
                "status": "ready",
                "message": "Application is ready to serve requests",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "not_ready",
                "message": "Application is not ready",
                "database": db_status["status"],
                "dependencies": deps_status["status"],
                "timestamp": datetime.now().isoformat()
            }


@click.command()
@click.option("--check-type", "-t", 
              type=click.Choice(["full", "liveness", "readiness"]),
              default="full",
              help="Type of health check to perform")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def health_check(check_type: str, output_json: bool) -> None:
    """Perform health check for deployment monitoring.
    
    Examples:
        ceramic-discovery health --check-type liveness
        ceramic-discovery health --check-type readiness --json
        ceramic-discovery health --check-type full
    """
    checker = HealthChecker()
    
    if check_type == "liveness":
        result = checker.liveness_check()
    elif check_type == "readiness":
        result = checker.readiness_check()
    else:
        result = checker.full_health_check()
    
    if output_json:
        import json
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


def _print_health_result(result: Dict[str, Any], check_type: str) -> None:
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
    health_check()
