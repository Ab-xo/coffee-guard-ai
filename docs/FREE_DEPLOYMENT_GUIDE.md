# CoffeeGuard AI - Free Open-Source Deployment Guide

**Version**: 1.0  
**Cost**: $0/month (100% FREE)  
**Best For**: Students, researchers, open-source projects, demos

---

## 🆓 100% Free Deployment Options

All platforms below offer **permanent free tiers** - no credit card required, no time limits!

---

## Option 1: Hugging Face Spaces (Recommended - EASIEST!)

### Why Hugging Face?
- ✅ **100% FREE forever** (no credit card needed)
- ✅ **Perfect for ML projects** - built for AI/ML apps
- ✅ **Automatic HTTPS**
- ✅ **Great community** - 500k+ ML practitioners
- ✅ **Easy GitHub integration**
- ✅ **2 vCPU + 16GB RAM** on free tier

### Setup Time: 10 minutes

### Step 1: Create Account
1. Go to https://huggingface.co/join
2. Sign up (free, no credit card)
3. Verify email

### Step 2: Create Streamlit Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Configure:
   - **Space name**: `coffeeguard-ai`
   - **License**: MIT
   - **SDK**: Streamlit
   - **Hardware**: CPU (free)
   - **Visibility**: Public

### Step 3: Configure Space

Create these files in your Space:

**`app.py`** (main app file):
```python
import streamlit as st
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import your Streamlit app
from apps.web.streamlit_app import main

if __name__ == "__main__":
    main()
```

**`requirements.txt`**:
```txt
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
onnxruntime>=1.16.0
pillow>=10.0.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
streamlit>=1.28.0
fastapi>=0.104.0
pydantic>=2.0.0
```

**`README.md`**:
```markdown
---
title: CoffeeGuard AI
emoji: ☕
colorFrom: green
colorTo: brown
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

# CoffeeGuard AI

Ethiopian coffee leaf disease detection system. Upload a coffee leaf image to detect:
- Healthy
- Cercospora (Brown Eye Spot)
- Leaf Rust
- Phoma

Built with EfficientNetV2-B0. See [GitHub](https://github.com/Ab-xo/coffee-guard-ai) for details.
```

### Step 4: Upload Files

**Option A - Git (Recommended):**
```bash
# Clone your Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/coffeeguard-ai
cd coffeeguard-ai

# Copy files from your project
cp -r ../coffee-guard-ai/src .
cp -r ../coffee-guard-ai/apps .
cp -r ../coffee-guard-ai/artifacts .
cp ../coffee-guard-ai/pyproject.toml .

# Add files
git add .
git commit -m "Initial deployment"
git push
```

**Option B - Web UI:**
1. Click "Files" tab
2. Click "Add file" → "Upload files"
3. Upload: `src/`, `apps/`, `artifacts/`, `requirements.txt`, `app.py`

### Step 5: Done! ✅

Your app will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/coffeeguard-ai
```

**Free tier includes:**
- Automatic HTTPS
- Permanent hosting
- Auto-restart on crashes
- Usage analytics
- Community feedback

---

## Option 2: Streamlit Community Cloud

### Why Streamlit Cloud?
- ✅ **Made for Streamlit apps** (perfect fit!)
- ✅ **1GB RAM, 1 vCPU** free forever
- ✅ **Direct GitHub deployment**
- ✅ **No configuration needed**
- ✅ **Custom domain support**

### Setup Time: 5 minutes

### Steps:

1. **Sign up**: https://share.streamlit.io/signup (use GitHub)

2. **Deploy**:
   - Click "New app"
   - Repository: `Ab-xo/coffee-guard-ai`
   - Branch: `main`
   - Main file path: `apps/web/streamlit_app.py`
   - Click "Deploy"

3. **Environment Setup**:
   - Add secret in "Advanced settings" → "Secrets":
   ```toml
   [general]
   API_URL = "http://localhost:8000"  # We'll fix this next
   ```

4. **Note**: Streamlit Cloud only hosts the web UI. For the API, use one of:
   - **Render.com** (API free tier)
   - **Fly.io** (see below)
   - **Run API locally** (for testing)

Your app will be live at:
```
https://coffeeguard-ai-YOUR_USERNAME.streamlit.app
```

---

## Option 3: Fly.io (Full Stack - API + Web)

### Why Fly.io?
- ✅ **Real Docker deployment** (most flexible)
- ✅ **3 shared-cpu VMs free** 
- ✅ **160GB data transfer/month** free
- ✅ **Global edge network**
- ✅ **Easy scaling** when needed

### Setup Time: 15 minutes

### Step 1: Install Fly CLI

**Windows (PowerShell):**
```powershell
iwr https://fly.io/install.ps1 -useb | iex
```

**Mac/Linux:**
```bash
curl -L https://fly.io/install.sh | sh
```

### Step 2: Sign Up & Login
```bash
fly auth signup  # Free account, no credit card
fly auth login
```

### Step 3: Deploy API

**Create `fly.api.toml`:**
```toml
app = "coffeeguard-api"

[build]
  dockerfile = "Dockerfile.api"

[env]
  MODEL_BUNDLE = "/app/artifacts/models/coffeeguard-effv2b0-v1"
  LOG_LEVEL = "INFO"

[[services]]
  internal_port = 8000
  protocol = "tcp"

  [[services.ports]]
    port = 80
    handlers = ["http"]
  
  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

  [[services.http_checks]]
    interval = "10s"
    timeout = "2s"
    grace_period = "30s"
    method = "GET"
    path = "/health"

[mounts]
  source = "coffeeguard_data"
  destination = "/data"
```

**Deploy API:**
```bash
fly launch --config fly.api.toml --name coffeeguard-api
fly deploy --config fly.api.toml
```

### Step 4: Deploy Web UI

**Create `fly.web.toml`:**
```toml
app = "coffeeguard-web"

[build]
  dockerfile = "Dockerfile.web"

[env]
  API_URL = "https://coffeeguard-api.fly.dev"

[[services]]
  internal_port = 8501
  protocol = "tcp"

  [[services.ports]]
    port = 80
    handlers = ["http"]
  
  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]
```

**Deploy Web:**
```bash
fly launch --config fly.web.toml --name coffeeguard-web
fly deploy --config fly.web.toml
```

### Step 5: Access Your App ✅

- **API**: https://coffeeguard-api.fly.dev
- **Web**: https://coffeeguard-web.fly.dev

**Free tier includes:**
- 3 shared-cpu VMs (256MB RAM each)
- 3GB persistent storage
- 160GB data transfer/month
- Automatic SSL

---

## Option 4: Koyeb (Serverless Free Tier)

### Why Koyeb?
- ✅ **Serverless architecture** (auto-sleep when idle)
- ✅ **512MB RAM, 2GB disk** free
- ✅ **GitHub auto-deploy**
- ✅ **Global edge locations**

### Setup Time: 10 minutes

### Steps:

1. **Sign up**: https://app.koyeb.com/signup (free, GitHub login)

2. **Create App**:
   - Click "Create App"
   - Source: GitHub repository
   - Repository: `Ab-xo/coffee-guard-ai`
   - Branch: `main`

3. **Configure API Service**:
   - Name: `coffeeguard-api`
   - Build: Docker
   - Dockerfile: `Dockerfile.api`
   - Port: 8000
   - Environment variables:
     ```
     MODEL_BUNDLE=artifacts/models/coffeeguard-effv2b0-v1
     LOG_LEVEL=INFO
     ```

4. **Configure Web Service**:
   - Name: `coffeeguard-web`
   - Build: Docker
   - Dockerfile: `Dockerfile.web`
   - Port: 8501
   - Environment variables:
     ```
     API_URL=https://coffeeguard-api-YOUR_ORG.koyeb.app
     ```

5. **Deploy** ✅

Your apps will be at:
- API: `https://coffeeguard-api-YOUR_ORG.koyeb.app`
- Web: `https://coffeeguard-web-YOUR_ORG.koyeb.app`

---

## Option 5: Render.com (Generous Free Tier)

### Why Render?
- ✅ **750 hours/month free** (enough for 1 always-on service)
- ✅ **Easy setup**
- ✅ **Automatic SSL**
- ✅ **Good documentation**

### Setup Time: 10 minutes

### Steps:

1. **Sign up**: https://render.com (use GitHub)

2. **Create Web Service** (for Streamlit):
   - New → Web Service
   - Connect `Ab-xo/coffee-guard-ai`
   - Name: `coffeeguard-web`
   - Environment: Python 3
   - Build Command: `pip install uv && uv sync`
   - Start Command: `uv run streamlit run apps/web/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
   - Plan: **Free**

3. **Environment Variables**:
   ```
   PYTHON_VERSION=3.12
   API_URL=http://localhost:8000
   ```

**Note**: Render free tier allows 1 web service. For both API + Web, one must be paid ($7/month) or use Hugging Face for one component.

---

## Option 6: Vercel + Serverless Functions

### Why Vercel?
- ✅ **100GB bandwidth/month** free
- ✅ **Serverless functions**
- ✅ **Best for web frontends**
- ✅ **Lightning fast**

### Setup Time: 20 minutes

### Steps:

1. **Install Vercel CLI**:
```bash
npm install -g vercel
```

2. **Login**:
```bash
vercel login
```

3. **Create `vercel.json`**:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "apps/api/app/main.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "apps/api/app/main.py"
    }
  ]
}
```

4. **Deploy**:
```bash
vercel --prod
```

**Limitation**: Serverless functions have 10s timeout on free tier (may be tight for model inference). Best for web UI + external API.

---

## Option 7: Self-Hosted on Free VPS

### Free VPS Providers:

#### Oracle Cloud (Always Free Tier)
- ✅ **4 ARM vCPUs + 24GB RAM** OR **2 AMD vCPUs + 1GB RAM**
- ✅ **Permanent free tier**
- ✅ **200GB storage**

**Steps:**
1. Sign up: https://www.oracle.com/cloud/free/
2. Create VM instance (Ubuntu 22.04)
3. Install Docker:
```bash
sudo apt update
sudo apt install docker.io docker-compose -y
```
4. Clone repo and run:
```bash
git clone https://github.com/Ab-xo/coffee-guard-ai.git
cd coffee-guard-ai
sudo docker-compose up -d
```
5. Open ports 8000 and 8501 in Oracle Cloud Network Security

#### Azure for Students
- ✅ **$100 credit** (lasts ~12 months)
- ✅ **Free services** for students
- Requirements: .edu email or student verification

#### AWS Free Tier
- ✅ **750 hours/month EC2 t2.micro** (12 months)
- ✅ Good for learning deployment

---

## Comparison Table

| Platform | API Support | Web UI | RAM | Storage | Bandwidth | Setup Time | Best For |
|----------|-------------|--------|-----|---------|-----------|------------|----------|
| **Hugging Face** | ❌ | ✅ | 16GB | 50GB | Unlimited | 10 min | **Easiest ML deployment** |
| **Streamlit Cloud** | ❌ | ✅ | 1GB | 1GB | Good | 5 min | **Streamlit apps only** |
| **Fly.io** | ✅ | ✅ | 768MB | 3GB | 160GB/mo | 15 min | **Full stack apps** |
| **Koyeb** | ✅ | ✅ | 512MB | 2GB | 100GB/mo | 10 min | **Serverless** |
| **Render** | ✅ | ✅ | 512MB | 1GB | 100GB/mo | 10 min | **Simple deployment** |
| **Oracle Cloud** | ✅ | ✅ | 24GB | 200GB | 10TB/mo | 30 min | **Maximum resources** |

---

## Recommended Strategy

### For Immediate Demo (5 minutes):
**🏆 Hugging Face Spaces**
- Easiest setup
- Perfect for ML projects
- Great for portfolio/demo

### For Full Production (15 minutes):
**🏆 Fly.io**
- Complete API + Web deployment
- Real production environment
- Easy to scale later

### For Maximum Resources (30 minutes):
**🏆 Oracle Cloud Always Free**
- 24GB RAM (best for model inference)
- No time limits
- Full VM control

### Hybrid Approach (10 minutes):
- **API**: Fly.io or Render
- **Web UI**: Hugging Face Spaces or Streamlit Cloud

---

## Quick Start: Hugging Face Deployment

Here's the fastest path to get your app live:

```bash
# 1. Create account on huggingface.co (30 seconds)

# 2. Create Space (2 minutes)
# https://huggingface.co/spaces → New Space → Streamlit

# 3. Clone and push (5 minutes)
git clone https://huggingface.co/spaces/YOUR_USERNAME/coffeeguard-ai
cd coffeeguard-ai

# Copy files
cp -r ../coffee-guard-ai/src .
cp -r ../coffee-guard-ai/apps .
cp -r ../coffee-guard-ai/artifacts .

# Create app.py
cat > app.py << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Run Streamlit app
import streamlit.web.cli as stcli
import sys

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "apps/web/streamlit_app.py"]
    sys.exit(stcli.main())
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
onnxruntime>=1.16.0
pillow>=10.0.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
streamlit>=1.28.0
fastapi>=0.104.0
pydantic>=2.0.0
EOF

# Push
git add .
git commit -m "Deploy CoffeeGuard AI"
git push

# 4. Done! Your app is live ✅
# https://huggingface.co/spaces/YOUR_USERNAME/coffeeguard-ai
```

---

## Troubleshooting

### Out of Memory on Free Tier
**Solution**: Use ONNX runtime (already implemented) - much lighter than PyTorch

### Slow Cold Starts
**Solution**: 
- Hugging Face: Keeps app warm if used regularly
- Fly.io: Use `min_machines_running = 1` in fly.toml

### Model Too Large
**Solution**: Model is already optimized (41MB). If needed:
```bash
# Quantize ONNX model (8-bit)
python -m onnxruntime.quantization.quantize_dynamic \
  artifacts/models/coffeeguard-effv2b0-v1/model.onnx \
  artifacts/models/coffeeguard-effv2b0-v1/model-int8.onnx
```

### Deployment Fails
**Check**:
1. requirements.txt includes all dependencies
2. Model files are included (not .gitignore'd)
3. Python version matches (3.12)
4. All paths are relative, not absolute

---

## Cost Comparison (Monthly)

| Platform | Free Tier | Paid (if needed) |
|----------|-----------|------------------|
| Hugging Face | ✅ **FREE** | N/A (always free) |
| Streamlit Cloud | ✅ **FREE** | N/A (always free) |
| Fly.io | ✅ **FREE** | $5-10 (scale up) |
| Koyeb | ✅ **FREE** | $7 (more resources) |
| Render | ✅ **FREE** | $7 per service |
| Oracle Cloud | ✅ **FREE** | Never needed |

---

## Security Notes

Even on free tiers, follow security best practices:

1. **HTTPS**: All platforms provide free SSL ✓
2. **Rate limiting**: Implement in code (already done)
3. **Input validation**: Already implemented ✓
4. **No secrets in code**: Use platform environment variables
5. **CORS**: Configure properly for your domain

---

## Monitoring (Free Tools)

### UptimeRobot
- https://uptimerobot.com
- Monitor uptime (50 monitors free)
- Email alerts when down

### Better Uptime
- https://betteruptime.com
- 10 monitors free
- Status page

### Sentry
- https://sentry.io
- Error tracking (5k events/month free)
- Performance monitoring

---

## Next Steps After Deployment

1. **Share your app**:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/coffeeguard-ai
   ```

2. **Add to portfolio/CV**:
   - "Deployed ML-powered disease detection system serving 1000+ farmers"
   - Include metrics: accuracy, uptime, usage stats

3. **Collect feedback**:
   - Add feedback button in Streamlit
   - Track usage with Google Analytics (free)

4. **Monitor usage**:
   - Hugging Face provides analytics dashboard
   - Track: daily users, countries, predictions

5. **Scale when needed**:
   - Free tiers handle 100-500 users/day
   - Upgrade only when you need more

---

## Support

### Hugging Face Community
- Forum: https://discuss.huggingface.co
- Discord: https://hf.co/join/discord

### Fly.io Community
- Forum: https://community.fly.io
- Discord: https://fly.io/discord

### Your Project
- GitHub Issues: https://github.com/Ab-xo/coffee-guard-ai/issues

---

## Conclusion

**Best FREE option for CoffeeGuard AI: Hugging Face Spaces**

✅ Perfect for ML projects  
✅ 16GB RAM (handles model easily)  
✅ Great for portfolio  
✅ Active community  
✅ No credit card needed  
✅ Setup in 10 minutes  

**Deploy now and start helping Ethiopian coffee farmers - for FREE! 🚀☕**
