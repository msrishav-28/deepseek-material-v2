"""Tests for containerized deployment and health checks."""

import json
import subprocess
import time
from pathlib import Path

import pytest


class TestHealthChecks:
    """Test health check functionality."""
    
    def test_health_module_imports(self):
        """Test that health module can be imported."""
        from ceramic_discovery.health import HealthChecker
        
        checker = HealthChecker()
        assert checker is not None
    
    def test_liveness_check(self):
        """Test liveness check returns healthy status."""
        from ceramic_discovery.health import HealthChecker
        
        checker = HealthChecker()
        result = checker.liveness_check()
        
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert result["message"] == "Application is running"
    
    def test_system_info(self):
        """Test system information collection."""
        from ceramic_discovery.health import HealthChecker
        
        checker = HealthChecker()
        info = checker.get_system_info()
        
        assert "python_version" in info
        assert "platform" in info
        assert "uptime_seconds" in info
        assert "timestamp" in info
        assert info["uptime_seconds"] >= 0
    
    def test_dependency_check(self):
        """Test dependency checking."""
        from ceramic_discovery.health import HealthChecker
        
        checker = HealthChecker()
        result = checker.check_dependencies()
        
        assert "status" in result
        assert "dependencies" in result
        
        # Check critical dependencies
        deps = result["dependencies"]
        assert "numpy" in deps
        assert "pandas" in deps
        assert "scikit-learn" in deps
        
        # Check that core dependencies are healthy
        core_deps = ["numpy", "pandas", "scikit-learn", "sqlalchemy"]
        for dep_name in core_deps:
            if dep_name in deps:
                dep_info = deps[dep_name]
                assert dep_info["status"] == "healthy", f"{dep_name} is not healthy"
                assert "version" in dep_info or "error" in dep_info
    
    def test_storage_check(self):
        """Test storage path checking."""
        from ceramic_discovery.health import HealthChecker
        
        checker = HealthChecker()
        result = checker.check_storage()
        
        assert "status" in result
        assert "paths" in result
        
        # Check that paths are reported
        paths = result["paths"]
        assert "hdf5" in paths or "results" in paths or "logs" in paths
    
    def test_api_keys_check(self):
        """Test API key configuration checking."""
        from ceramic_discovery.health import HealthChecker
        
        checker = HealthChecker()
        result = checker.check_api_keys()
        
        assert "status" in result
        assert "configured" in result
        assert "materials_project" in result["configured"]
    
    def test_full_health_check_structure(self):
        """Test full health check returns proper structure."""
        from ceramic_discovery.health import HealthChecker
        
        checker = HealthChecker()
        result = checker.full_health_check()
        
        assert "status" in result
        assert "checks" in result
        assert "system" in result
        
        # Check that all component checks are present
        checks = result["checks"]
        assert "database" in checks
        assert "redis" in checks
        assert "storage" in checks
        assert "dependencies" in checks
        assert "api_keys" in checks
        
        # System info should be present
        system = result["system"]
        assert "python_version" in system
        assert "uptime_seconds" in system


class TestDockerConfiguration:
    """Test Docker configuration files."""
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists."""
        dockerfile = Path("Dockerfile")
        assert dockerfile.exists()
    
    def test_dockerfile_has_healthcheck(self):
        """Test that Dockerfile includes health check."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        
        assert "HEALTHCHECK" in content
        assert "health" in content
        assert "liveness" in content
    
    def test_dockerfile_has_conda_env(self):
        """Test that Dockerfile sets up conda environment."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        
        assert "conda" in content
        assert "environment.yml" in content
        assert "ceramic-armor-discovery" in content
    
    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists."""
        compose_file = Path("docker-compose.yml")
        assert compose_file.exists()
    
    def test_docker_compose_has_services(self):
        """Test that docker-compose defines required services."""
        import yaml
        
        compose_file = Path("docker-compose.yml")
        with open(compose_file) as f:
            config = yaml.safe_load(f)
        
        assert "services" in config
        services = config["services"]
        
        # Check required services
        assert "postgres" in services
        assert "redis" in services
        assert "app" in services
    
    def test_docker_compose_has_healthchecks(self):
        """Test that docker-compose services have health checks."""
        import yaml
        
        compose_file = Path("docker-compose.yml")
        with open(compose_file) as f:
            config = yaml.safe_load(f)
        
        services = config["services"]
        
        # Check health checks
        assert "healthcheck" in services["postgres"]
        assert "healthcheck" in services["redis"]
        assert "healthcheck" in services["app"]
    
    def test_docker_compose_has_volumes(self):
        """Test that docker-compose defines persistent volumes."""
        import yaml
        
        compose_file = Path("docker-compose.yml")
        with open(compose_file) as f:
            config = yaml.safe_load(f)
        
        assert "volumes" in config
        volumes = config["volumes"]
        
        # Check required volumes
        assert "postgres_data" in volumes
        assert "redis_data" in volumes
        assert "hdf5_data" in volumes
        assert "results" in volumes


class TestKubernetesManifests:
    """Test Kubernetes deployment manifests."""
    
    def test_k8s_directory_exists(self):
        """Test that k8s directory exists."""
        k8s_dir = Path("k8s")
        assert k8s_dir.exists()
        assert k8s_dir.is_dir()
    
    def test_k8s_manifests_exist(self):
        """Test that all required K8s manifests exist."""
        k8s_dir = Path("k8s")
        
        required_files = [
            "namespace.yaml",
            "configmap.yaml",
            "secrets.yaml",
            "postgres-deployment.yaml",
            "redis-deployment.yaml",
            "app-deployment.yaml",
            "ingress.yaml",
            "kustomization.yaml"
        ]
        
        for filename in required_files:
            filepath = k8s_dir / filename
            assert filepath.exists(), f"Missing {filename}"
    
    def test_k8s_namespace_manifest(self):
        """Test namespace manifest structure."""
        import yaml
        
        manifest_file = Path("k8s/namespace.yaml")
        with open(manifest_file) as f:
            manifest = yaml.safe_load(f)
        
        assert manifest["kind"] == "Namespace"
        assert manifest["metadata"]["name"] == "ceramic-discovery"
    
    def test_k8s_configmap_manifest(self):
        """Test configmap manifest structure."""
        import yaml
        
        manifest_file = Path("k8s/configmap.yaml")
        with open(manifest_file) as f:
            manifest = yaml.safe_load(f)
        
        assert manifest["kind"] == "ConfigMap"
        assert "data" in manifest
        
        # Check key configuration values
        data = manifest["data"]
        assert "MAX_PARALLEL_JOBS" in data
        assert "LOG_LEVEL" in data
    
    def test_k8s_app_deployment_has_healthchecks(self):
        """Test that app deployment includes health checks."""
        import yaml
        
        manifest_file = Path("k8s/app-deployment.yaml")
        with open(manifest_file) as f:
            # Load all documents
            manifests = list(yaml.safe_load_all(f))
        
        # Find the Deployment manifest
        deployment = None
        for manifest in manifests:
            if manifest and manifest.get("kind") == "Deployment":
                deployment = manifest
                break
        
        assert deployment is not None
        
        # Check health probes
        containers = deployment["spec"]["template"]["spec"]["containers"]
        app_container = containers[0]
        
        assert "livenessProbe" in app_container
        assert "readinessProbe" in app_container
        
        # Check that health checks use our CLI
        liveness = app_container["livenessProbe"]
        assert "exec" in liveness
        assert "health" in str(liveness["exec"]["command"])
        assert "liveness" in str(liveness["exec"]["command"])
        
        readiness = app_container["readinessProbe"]
        assert "exec" in readiness
        assert "health" in str(readiness["exec"]["command"])
        assert "readiness" in str(readiness["exec"]["command"])
    
    def test_k8s_postgres_deployment_has_healthchecks(self):
        """Test that postgres deployment includes health checks."""
        import yaml
        
        manifest_file = Path("k8s/postgres-deployment.yaml")
        with open(manifest_file) as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Find the Deployment manifest
        deployment = None
        for manifest in manifests:
            if manifest and manifest.get("kind") == "Deployment":
                deployment = manifest
                break
        
        assert deployment is not None
        
        containers = deployment["spec"]["template"]["spec"]["containers"]
        postgres_container = containers[0]
        
        assert "livenessProbe" in postgres_container
        assert "readinessProbe" in postgres_container
    
    def test_k8s_redis_deployment_has_healthchecks(self):
        """Test that redis deployment includes health checks."""
        import yaml
        
        manifest_file = Path("k8s/redis-deployment.yaml")
        with open(manifest_file) as f:
            manifests = list(yaml.safe_load_all(f))
        
        # Find the Deployment manifest
        deployment = None
        for manifest in manifests:
            if manifest and manifest.get("kind") == "Deployment":
                deployment = manifest
                break
        
        assert deployment is not None
        
        containers = deployment["spec"]["template"]["spec"]["containers"]
        redis_container = containers[0]
        
        assert "livenessProbe" in redis_container
        assert "readinessProbe" in redis_container
    
    def test_k8s_readme_exists(self):
        """Test that K8s README exists."""
        readme = Path("k8s/README.md")
        assert readme.exists()
        
        content = readme.read_text()
        assert "Kubernetes Deployment" in content
        assert "Prerequisites" in content
        assert "Quick Start" in content


class TestCLIHealthCommand:
    """Test CLI health command."""
    
    def test_health_command_exists(self):
        """Test that health command is registered in CLI."""
        from ceramic_discovery.cli import main
        
        # Check that health command is in the CLI
        commands = [cmd.name for cmd in main.commands.values()]
        assert "health" in commands
    
    def test_health_command_has_options(self):
        """Test that health command has proper options."""
        from ceramic_discovery.cli import main
        
        health_cmd = main.commands["health"]
        
        # Check parameters
        param_names = [p.name for p in health_cmd.params]
        assert "check_type" in param_names
        assert "output_json" in param_names


class TestEnvironmentConfiguration:
    """Test environment configuration files."""
    
    def test_env_example_exists(self):
        """Test that .env.example exists."""
        env_example = Path(".env.example")
        assert env_example.exists()
    
    def test_environment_yml_exists(self):
        """Test that environment.yml exists."""
        env_yml = Path("environment.yml")
        assert env_yml.exists()
    
    def test_environment_yml_has_dependencies(self):
        """Test that environment.yml includes all required dependencies."""
        import yaml
        
        env_yml = Path("environment.yml")
        with open(env_yml) as f:
            config = yaml.safe_load(f)
        
        assert "dependencies" in config
        
        # Convert to string for easier checking
        deps_str = str(config["dependencies"])
        
        # Check critical dependencies
        assert "python" in deps_str
        assert "numpy" in deps_str
        assert "pandas" in deps_str
        assert "scikit-learn" in deps_str
        assert "pymatgen" in deps_str
        assert "sqlalchemy" in deps_str
        assert "redis" in deps_str or "redis-py" in deps_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
