# Kubernetes Deployment for Ceramic Armor Discovery Framework

This directory contains Kubernetes manifests for deploying the Ceramic Armor Discovery Framework in a production environment.

## Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured to access your cluster
- Docker registry access (for custom images)
- Storage provisioner for PersistentVolumes
- (Optional) Ingress controller (nginx recommended)
- (Optional) cert-manager for TLS certificates

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -t your-registry/ceramic-discovery:latest .

# Push to your registry
docker push your-registry/ceramic-discovery:latest
```

### 2. Update Configuration

Edit `secrets.yaml` to set your credentials:
```yaml
POSTGRES_PASSWORD: your-secure-password
MATERIALS_PROJECT_API_KEY: your-api-key
DATABASE_URL: postgresql://ceramic_user:your-secure-password@postgres:5432/ceramic_materials
```

**Important**: In production, use proper secret management (e.g., sealed-secrets, external-secrets, or cloud provider secret managers).

Edit `configmap.yaml` to adjust application settings as needed.

Edit `ingress.yaml` to set your domain name:
```yaml
- host: your-domain.com
```

### 3. Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -k .

# Or apply individually
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml
kubectl apply -f app-deployment.yaml
kubectl apply -f ingress.yaml
```

### 4. Verify Deployment

```bash
# Check all resources
kubectl get all -n ceramic-discovery

# Check pod status
kubectl get pods -n ceramic-discovery

# Check health
kubectl exec -n ceramic-discovery deployment/ceramic-discovery-app -- \
  conda run -n ceramic-armor-discovery python -m ceramic_discovery.cli health --check-type full

# View logs
kubectl logs -n ceramic-discovery deployment/ceramic-discovery-app -f
```

## Architecture

The deployment consists of:

- **PostgreSQL**: Database with pgvector extension for material data
- **Redis**: Cache for screening results and ML predictions
- **Application**: Main Ceramic Discovery Framework with Jupyter Lab

### Resource Requirements

Default resource allocations:

- **PostgreSQL**: 2-4 GiB RAM, 1-2 CPU cores, 50 GiB storage
- **Redis**: 1-2 GiB RAM, 0.5-1 CPU cores, 10 GiB storage
- **Application**: 4-8 GiB RAM, 2-4 CPU cores, 150 GiB storage (HDF5 + results)

Adjust in the deployment files based on your workload.

## Storage

The deployment uses PersistentVolumeClaims for:

- `postgres-pvc`: PostgreSQL data (50 GiB)
- `redis-pvc`: Redis persistence (10 GiB)
- `hdf5-data-pvc`: HDF5 computational data (100 GiB, ReadWriteMany)
- `results-pvc`: Analysis results (50 GiB, ReadWriteMany)

Ensure your cluster has a storage provisioner that supports these access modes.

## Health Checks

The application includes comprehensive health checks:

- **Liveness Probe**: Checks if the application is running
- **Readiness Probe**: Checks if the application is ready to serve requests

Health check endpoints:
```bash
# Liveness check
kubectl exec -n ceramic-discovery deployment/ceramic-discovery-app -- \
  conda run -n ceramic-armor-discovery python -m ceramic_discovery.cli health --check-type liveness

# Readiness check
kubectl exec -n ceramic-discovery deployment/ceramic-discovery-app -- \
  conda run -n ceramic-armor-discovery python -m ceramic_discovery.cli health --check-type readiness

# Full health check
kubectl exec -n ceramic-discovery deployment/ceramic-discovery-app -- \
  conda run -n ceramic-armor-discovery python -m ceramic_discovery.cli health --check-type full
```

## Scaling

### Horizontal Scaling

The application deployment can be scaled horizontally for read-heavy workloads:

```bash
kubectl scale deployment ceramic-discovery-app -n ceramic-discovery --replicas=3
```

**Note**: Ensure your storage supports ReadWriteMany for multiple replicas.

### Vertical Scaling

Adjust resource limits in `app-deployment.yaml`:

```yaml
resources:
  requests:
    memory: "8Gi"
    cpu: "4000m"
  limits:
    memory: "16Gi"
    cpu: "8000m"
```

## Monitoring

### View Logs

```bash
# Application logs
kubectl logs -n ceramic-discovery deployment/ceramic-discovery-app -f

# PostgreSQL logs
kubectl logs -n ceramic-discovery deployment/postgres -f

# Redis logs
kubectl logs -n ceramic-discovery deployment/redis -f
```

### Port Forwarding

Access services locally:

```bash
# Jupyter Lab
kubectl port-forward -n ceramic-discovery svc/ceramic-discovery-app 8888:8888

# PostgreSQL
kubectl port-forward -n ceramic-discovery svc/postgres 5432:5432

# Redis
kubectl port-forward -n ceramic-discovery svc/redis 6379:6379
```

## Backup and Recovery

### Database Backup

```bash
# Backup PostgreSQL
kubectl exec -n ceramic-discovery deployment/postgres -- \
  pg_dump -U ceramic_user ceramic_materials > backup.sql

# Restore PostgreSQL
kubectl exec -i -n ceramic-discovery deployment/postgres -- \
  psql -U ceramic_user ceramic_materials < backup.sql
```

### Data Backup

```bash
# Backup HDF5 data
kubectl cp ceramic-discovery/ceramic-discovery-app:/opt/ceramic-armor-discovery/data/hdf5 ./hdf5-backup

# Backup results
kubectl cp ceramic-discovery/ceramic-discovery-app:/opt/ceramic-armor-discovery/results ./results-backup
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod -n ceramic-discovery <pod-name>

# Check logs
kubectl logs -n ceramic-discovery <pod-name>
```

### Database Connection Issues

```bash
# Test database connectivity
kubectl exec -n ceramic-discovery deployment/ceramic-discovery-app -- \
  conda run -n ceramic-armor-discovery python -c "from ceramic_discovery.dft.database_manager import DatabaseManager; db = DatabaseManager(); print('Connected')"
```

### Storage Issues

```bash
# Check PVC status
kubectl get pvc -n ceramic-discovery

# Check PV status
kubectl get pv
```

### Health Check Failures

```bash
# Run manual health check
kubectl exec -n ceramic-discovery deployment/ceramic-discovery-app -- \
  conda run -n ceramic-armor-discovery python -m ceramic_discovery.cli health --check-type full --json
```

## Security Considerations

1. **Secrets Management**: Use proper secret management solutions in production
2. **Network Policies**: Implement network policies to restrict pod-to-pod communication
3. **RBAC**: Configure appropriate role-based access control
4. **TLS**: Enable TLS for all external communications
5. **Image Security**: Scan images for vulnerabilities before deployment

## Cleanup

Remove all resources:

```bash
# Delete all resources
kubectl delete -k .

# Or delete namespace (removes everything)
kubectl delete namespace ceramic-discovery
```

## Production Checklist

- [ ] Update secrets with secure credentials
- [ ] Configure proper storage class and sizes
- [ ] Set up backup strategy
- [ ] Configure monitoring and alerting
- [ ] Set up log aggregation
- [ ] Configure network policies
- [ ] Enable TLS/SSL
- [ ] Set resource limits appropriately
- [ ] Configure autoscaling if needed
- [ ] Set up disaster recovery plan
- [ ] Document runbooks for common operations
