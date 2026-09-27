# CoffeeGuard AI - Production Deployment Guide

**Version**: 1.1.0  
**Last Updated**: September 2026  
**Status**: Production Ready

---

## Overview

This guide covers deploying CoffeeGuard AI to production using industry-standard platforms. The system consists of:

- **FastAPI Backend** (disease detection API)
- **Streamlit Web App** (user interface)
- **ONNX Model** (efficient inference)
- **Static Assets** (sample images, documentation)

---

## Recommended Deployment Architecture

```
┌──────────────────────────────────────────────────────┐
│                    INTERNET                          │
└────────────────────┬─────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │   Load Balancer     │
          │   (HTTPS/SSL)       │
          └──────────┬──────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌───────────────┐
│   FastAPI     │         │   Streamlit   │
│   Container   │◄────────│   Container   │
│   (Port 8000) │         │   (Port 8501) │
└───────┬───────┘         └───────────────┘
        │
        ▼
┌───────────────┐
│  ONNX Model   │
│  (artifacts/) │
└───────────────┘
```

---

## Option 1: AWS Deployment (Recommended for Scale)

### Why AWS?
- ✅ Reliable, global infrastructure
- ✅ Auto-scaling capabilities
- ✅ Ethiopian data center (Bahir Dar) coming soon
- ✅ 12 months free tier for new accounts
- ✅ Easy to scale from 10 to 10,000+ users

### Architecture: AWS ECS Fargate + Application Load Balancer

#### Step 1: Prepare Docker Images

**1. Build and test locally:**

```bash
# Build API image
docker build -f Dockerfile.api -t coffeeguard-api:latest .

# Build Web image
docker build -f Dockerfile.web -t coffeeguard-web:latest .

# Test locally
docker-compose up
# Visit: http://localhost:8501
```

**2. Create AWS ECR repositories:**

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter: AWS Access Key ID, Secret Access Key, Region (af-south-1 for Cape Town)

# Create repositories
aws ecr create-repository --repository-name coffeeguard-api
aws ecr create-repository --repository-name coffeeguard-web

# Get login token
aws ecr get-login-password --region af-south-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.af-south-1.amazonaws.com

# Tag images
docker tag coffeeguard-api:latest YOUR_ACCOUNT_ID.dkr.ecr.af-south-1.amazonaws.com/coffeeguard-api:latest
docker tag coffeeguard-web:latest YOUR_ACCOUNT_ID.dkr.ecr.af-south-1.amazonaws.com/coffeeguard-web:latest

# Push to ECR
docker push YOUR_ACCOUNT_ID.dkr.ecr.af-south-1.amazonaws.com/coffeeguard-api:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.af-south-1.amazonaws.com/coffeeguard-web:latest
```

#### Step 2: Create ECS Cluster

**Using AWS Console:**

1. Go to **ECS** → **Clusters** → **Create Cluster**
2. Select **Networking only** (Fargate)
3. Cluster name: `coffeeguard-production`
4. Enable Container Insights (for monitoring)
5. Create

**Using AWS CLI:**

```bash
aws ecs create-cluster --cluster-name coffeeguard-production
```

#### Step 3: Create Task Definitions

**API Task Definition** (`ecs-task-api.json`):

```json
{
  "family": "coffeeguard-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.af-south-1.amazonaws.com/coffeeguard-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "MODEL_BUNDLE", "value": "/app/artifacts/models/coffeeguard-effv2b0-v1"},
        {"name": "LOG_LEVEL", "value": "INFO"},
        {"name": "MAX_UPLOAD_MB", "value": "10"},
        {"name": "CORS_ORIGINS", "value": "*"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/coffeeguard-api",
          "awslogs-region": "af-south-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

**Web Task Definition** (`ecs-task-web.json`):

```json
{
  "family": "coffeeguard-web",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "web",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.af-south-1.amazonaws.com/coffeeguard-web:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "API_URL", "value": "http://internal-alb.coffeeguard.local:8000"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/coffeeguard-web",
          "awslogs-region": "af-south-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Register tasks:**

```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-api.json
aws ecs register-task-definition --cli-input-json file://ecs-task-web.json
```

#### Step 4: Create Application Load Balancer

1. Go to **EC2** → **Load Balancers** → **Create Load Balancer**
2. Choose **Application Load Balancer**
3. Name: `coffeeguard-alb`
4. Scheme: **Internet-facing**
5. IP address type: **IPv4**
6. Network mapping: Select VPC and at least 2 availability zones
7. Security group: Allow HTTP (80) and HTTPS (443) from 0.0.0.0/0
8. Create target groups:
   - **api-target**: Port 8000, Health check path `/health`
   - **web-target**: Port 8501, Health check path `/`
9. Configure routing rules:
   - `/api/*` → api-target
   - `/` → web-target

#### Step 5: Create ECS Services

**API Service:**

```bash
aws ecs create-service \
  --cluster coffeeguard-production \
  --service-name api \
  --task-definition coffeeguard-api \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/api-target/...,containerName=api,containerPort=8000" \
  --health-check-grace-period-seconds 60
```

**Web Service:**

```bash
aws ecs create-service \
  --cluster coffeeguard-production \
  --service-name web \
  --task-definition coffeeguard-web \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/web-target/...,containerName=web,containerPort=8501" \
  --health-check-grace-period-seconds 60
```

#### Step 6: Set Up Auto Scaling

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/coffeeguard-production/api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10

# Create scaling policy (target 70% CPU)
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id service/coffeeguard-production/api \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-name cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

**scaling-policy.json:**
```json
{
  "TargetValue": 70.0,
  "PredefinedMetricSpecification": {
    "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
  },
  "ScaleOutCooldown": 60,
  "ScaleInCooldown": 300
}
```

#### Step 7: Configure Domain & SSL

1. **Register domain** (e.g., coffeeguard.et via Ethio Telecom or international registrar)
2. **Create hosted zone** in Route 53
3. **Request SSL certificate** in AWS Certificate Manager for `coffeeguard.et` and `*.coffeeguard.et`
4. **Add HTTPS listener** to ALB with certificate
5. **Create DNS records**:
   - `A` record: `coffeeguard.et` → ALB DNS
   - `CNAME`: `www.coffeeguard.et` → `coffeeguard.et`
   - `CNAME`: `api.coffeeguard.et` → ALB DNS

#### Cost Estimate (AWS Africa - Cape Town)

| Service | Configuration | Monthly Cost (USD) |
|---------|--------------|-------------------|
| ECS Fargate | 2 API tasks (1 vCPU, 2GB) | ~$70 |
| ECS Fargate | 2 Web tasks (0.5 vCPU, 1GB) | ~$35 |
| Application Load Balancer | Standard | ~$23 |
| Data Transfer | 100GB out | ~$9 |
| CloudWatch Logs | 5GB | ~$3 |
| ECR Storage | 2GB | ~$0.20 |
| **Total** | | **~$140/month** |

*Free tier: 12 months free for new accounts (limited usage)*

---

## Option 2: Railway.app (Fastest Deployment - 5 minutes)

### Why Railway?
- ✅ **Easiest deployment** (connects to GitHub, auto-deploys)
- ✅ **$5/month free tier** (enough for testing)
- ✅ **Automatic HTTPS**
- ✅ **Built-in monitoring**
- ❌ No Ethiopian servers (US/EU only)

### Deployment Steps

1. **Sign up**: https://railway.app (use GitHub account)

2. **Create new project** → **Deploy from GitHub repo**

3. **Select repository**: `Ab-xo/coffee-guard-ai`

4. **Add services**:
   - Service 1: API
     - **Build command**: `uv sync`
     - **Start command**: `uv run uvicorn app.main:app --app-dir apps/api --host 0.0.0.0 --port $PORT`
     - **Environment variables**:
       ```
       MODEL_BUNDLE=artifacts/models/coffeeguard-effv2b0-v1
       LOG_LEVEL=INFO
       CORS_ORIGINS=*
       ```
   
   - Service 2: Web
     - **Build command**: `uv sync`
     - **Start command**: `uv run streamlit run apps/web/app.py --server.port $PORT --server.address 0.0.0.0`
     - **Environment variables**:
       ```
       API_URL=${{API.RAILWAY_PRIVATE_DOMAIN}}
       ```

5. **Deploy**: Railway automatically builds and deploys!

6. **Get URLs**:
   - API: `https://coffeeguard-api-production.up.railway.app`
   - Web: `https://coffeeguard-web-production.up.railway.app`

7. **Add custom domain** (optional):
   - Settings → Domains → Add Custom Domain
   - Point `coffeeguard.et` CNAME to Railway domain

**Cost**: $5/month (free tier) → $20/month (Pro plan with more resources)

---

## Option 3: Render.com (Best Balance)

### Why Render?
- ✅ Simple deployment (like Railway)
- ✅ **Free tier** for static sites
- ✅ **$7/month** for web services
- ✅ Automatic SSL
- ✅ Good for Ethiopian users (global CDN)

### Deployment Steps

1. **Sign up**: https://render.com

2. **Create Blueprint** (`render.yaml`):

```yaml
services:
  - type: web
    name: coffeeguard-api
    env: python
    buildCommand: "uv sync"
    startCommand: "uv run uvicorn app.main:app --app-dir apps/api --host 0.0.0.0 --port $PORT"
    plan: starter
    envVars:
      - key: MODEL_BUNDLE
        value: artifacts/models/coffeeguard-effv2b0-v1
      - key: LOG_LEVEL
        value: INFO
      - key: PYTHON_VERSION
        value: "3.12"
    healthCheckPath: /health

  - type: web
    name: coffeeguard-web
    env: python
    buildCommand: "uv sync"
    startCommand: "uv run streamlit run apps/web/app.py --server.port $PORT --server.address 0.0.0.0"
    plan: starter
    envVars:
      - key: API_URL
        fromService:
          name: coffeeguard-api
          type: web
          property: host
      - key: PYTHON_VERSION
        value: "3.12"
```

3. **Connect GitHub repo** and deploy

4. **Access**:
   - API: `https://coffeeguard-api.onrender.com`
   - Web: `https://coffeeguard-web.onrender.com`

**Cost**: $7/month per service = $14/month total

---

## Option 4: Google Cloud Run (Serverless, Pay-per-use)

### Why Cloud Run?
- ✅ **Only pay when used** (can be $0-5/month for low traffic)
- ✅ Auto-scales to zero
- ✅ Global network
- ✅ $300 free credit for new accounts

### Deployment Steps

1. **Install gcloud CLI**: https://cloud.google.com/sdk/docs/install

2. **Initialize**:
```bash
gcloud init
gcloud config set project coffeeguard-ai
```

3. **Build and push images**:
```bash
# Enable services
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build with Cloud Build
gcloud builds submit --tag gcr.io/coffeeguard-ai/api -f Dockerfile.api
gcloud builds submit --tag gcr.io/coffeeguard-ai/web -f Dockerfile.web
```

4. **Deploy services**:
```bash
# Deploy API
gcloud run deploy coffeeguard-api \
  --image gcr.io/coffeeguard-ai/api \
  --platform managed \
  --region africa-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --set-env-vars MODEL_BUNDLE=artifacts/models/coffeeguard-effv2b0-v1

# Deploy Web
gcloud run deploy coffeeguard-web \
  --image gcr.io/coffeeguard-ai/web \
  --platform managed \
  --region africa-south1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --set-env-vars API_URL=https://coffeeguard-api-xxx-uc.a.run.app
```

5. **Map custom domain**:
```bash
gcloud run domain-mappings create --service coffeeguard-web --domain coffeeguard.et
```

**Cost**: ~$0-10/month (depends on usage)

---

## Option 5: DigitalOcean App Platform

### Why DigitalOcean?
- ✅ Simple, predictable pricing
- ✅ Good documentation
- ✅ $200 free credit
- ✅ Solid performance

### Deployment Steps

1. **Sign up**: https://www.digitalocean.com

2. **Create App** → **Deploy from GitHub**

3. **Configure**:
   - **API Component**:
     - Type: Web Service
     - Source: GitHub repo
     - Build Command: `uv sync`
     - Run Command: `uv run uvicorn app.main:app --app-dir apps/api --host 0.0.0.0 --port 8080`
     - Plan: Basic ($12/month)
   
   - **Web Component**:
     - Type: Web Service
     - Build Command: `uv sync`
     - Run Command: `uv run streamlit run apps/web/app.py --server.port 8080`
     - Plan: Basic ($5/month)

4. **Add domain** in settings

**Cost**: ~$17/month

---

## Comparison Table

| Platform | Ease of Setup | Cost/Month | Ethiopian Servers | Auto-Scale | Free Tier |
|----------|--------------|------------|-------------------|------------|-----------|
| **AWS** | ⭐⭐⭐ (Complex) | $140 | ❌ (Coming soon) | ✅ | ✅ (12 months) |
| **Railway** | ⭐⭐⭐⭐⭐ (Easiest) | $20 | ❌ | ✅ | ✅ ($5/month) |
| **Render** | ⭐⭐⭐⭐⭐ (Easy) | $14 | ❌ | ✅ | ✅ (Static sites) |
| **Cloud Run** | ⭐⭐⭐⭐ (Moderate) | $0-10 | ❌ | ✅✅ | ✅ ($300 credit) |
| **DigitalOcean** | ⭐⭐⭐⭐ (Easy) | $17 | ❌ | ✅ | ✅ ($200 credit) |

---

## Recommendation by Use Case

### For Testing/Demo (100 users/day)
**🏆 Railway.app** or **Render.com**
- Fastest setup (5 minutes)
- Free tier sufficient
- Easy to show to stakeholders

### For Production (1,000+ users/day)
**🏆 AWS ECS Fargate**
- Enterprise-grade reliability
- Auto-scaling
- Best for long-term growth

### For Cost-Effective Production (500 users/day)
**🏆 Google Cloud Run**
- Pay only for actual usage
- Can be very cheap for moderate traffic
- Good performance

### For Ethiopian Market (Future)
**🏆 AWS** - Wait for Bahir Dar data center (announced, not yet operational)

---

## Monitoring & Observability

### Health Checks

**API Health Endpoint**:
```bash
curl https://your-domain.com/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "EfficientNetV2-B0",
  "model_version": "1.1.0"
}
```

### Logging

**View logs (Railway)**:
```
Dashboard → Service → Logs tab
```

**View logs (AWS)**:
```bash
aws logs tail /ecs/coffeeguard-api --follow
```

**View logs (Cloud Run)**:
```bash
gcloud run services logs read coffeeguard-api --limit 50
```

### Metrics to Monitor

- **Request latency**: < 200ms p95
- **Error rate**: < 1%
- **CPU usage**: < 70%
- **Memory usage**: < 80%
- **Prediction accuracy**: Track via field feedback

### Alerting

Set up alerts for:
- Service down (health check failing)
- High error rate (>5%)
- High latency (>1s p95)
- High memory usage (>90%)

---

## Security Best Practices

### 1. Environment Variables
Never commit secrets to Git. Use platform secret management:

**Railway**:
```
Settings → Variables → Add Variable (encrypted)
```

**AWS**:
```bash
aws secretsmanager create-secret --name coffeeguard/api-key --secret-string "your-key"
```

### 2. HTTPS Only
Always enforce HTTPS in production:

```python
# In FastAPI main.py
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

if os.getenv("ENVIRONMENT") == "production":
    app.add_middleware(HTTPSRedirectMiddleware)
```

### 3. Rate Limiting
Prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("30/minute")
async def predict(...):
    ...
```

### 4. Input Validation
Already implemented in the codebase:
- File size limits (10MB)
- File type validation (JPEG/PNG only)
- Image dimension checks

### 5. CORS Configuration
For production, restrict origins:

```python
# Only allow your frontend domain
CORS_ORIGINS = ["https://coffeeguard.et", "https://www.coffeeguard.et"]
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install uv
        run: pip install uv
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        run: uv run pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      # Deploy to Railway (automatic with Railway GitHub integration)
      # OR Deploy to AWS:
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: af-south-1
      
      - name: Build and push Docker images
        run: |
          docker build -f Dockerfile.api -t coffeeguard-api .
          docker tag coffeeguard-api:latest ${{ secrets.ECR_REGISTRY }}/coffeeguard-api:latest
          docker push ${{ secrets.ECR_REGISTRY }}/coffeeguard-api:latest
      
      - name: Update ECS service
        run: |
          aws ecs update-service --cluster coffeeguard-production --service api --force-new-deployment
```

---

## Backup & Disaster Recovery

### Model Backup
- Model files are in Git (`artifacts/models/`)
- Additional backup to S3:
```bash
aws s3 sync artifacts/models/ s3://coffeeguard-models-backup/
```

### Database Backup (if you add one later)
- Daily automated backups
- Point-in-time recovery enabled
- Cross-region replication for critical data

### Disaster Recovery Plan
1. **RPO** (Recovery Point Objective): 1 hour
2. **RTO** (Recovery Time Objective): 15 minutes
3. **Failover**: Use multi-region deployment for critical workloads

---

## Scaling Strategy

### Phase 1: 0-1,000 users/day
- Single region deployment
- 2 API instances, 1 Web instance
- Cost: $15-50/month

### Phase 2: 1,000-10,000 users/day
- Multi-availability zone
- Auto-scaling (2-10 instances)
- CDN for static assets
- Cost: $100-300/month

### Phase 3: 10,000+ users/day
- Multi-region deployment
- Edge caching
- Dedicated databases
- Advanced monitoring
- Cost: $500-2,000/month

---

## Support & Maintenance

### Regular Tasks
- **Weekly**: Check error logs, review metrics
- **Monthly**: Update dependencies, security patches
- **Quarterly**: Review costs, optimize resources

### Monitoring Checklist
- [ ] All services healthy
- [ ] Response time < 200ms
- [ ] Error rate < 1%
- [ ] No security alerts
- [ ] Costs within budget

---

## Quick Start: Deploy in 5 Minutes

**For immediate demo deployment, use Railway:**

```bash
# 1. Push to GitHub (already done)
git push origin main

# 2. Go to railway.app
# 3. Click "Deploy from GitHub"
# 4. Select repository
# 5. Done! ✅
```

Your app will be live at:
- API: `https://coffeeguard-api-production.up.railway.app`
- Web: `https://coffeeguard-web-production.up.railway.app`

---

## Troubleshooting

### Service won't start
- Check logs for errors
- Verify MODEL_BUNDLE path is correct
- Ensure model files are in image

### High latency
- Check instance CPU/memory
- Scale up instance size
- Add CDN caching

### OOM (Out of Memory)
- Increase memory allocation (2GB minimum for API)
- Check for memory leaks in logs

### SSL certificate issues
- Verify domain DNS propagation
- Check certificate expiration
- Renew certificates before expiry

---

## Conclusion

**For fastest deployment RIGHT NOW**: Use **Railway.app** - you'll be live in 5 minutes.

**For production with Ethiopian users**: Use **AWS** and monitor for Bahir Dar data center launch.

**For cost-effective production**: Use **Google Cloud Run** - pay only for what you use.

All platforms are production-ready and will serve CoffeeGuard reliably to thousands of Ethiopian coffee farmers! 🚀☕

---

**Questions?**  
- Technical: github.com/Ab-xo/coffee-guard-ai/issues
- Email: [your-email]
