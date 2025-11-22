# 🚀 Deploy to Hugging Face Spaces

## Quick Deploy (5 minutes)

### 1. Create Hugging Face Account
- Go to https://huggingface.co/join
- Sign up (free)

### 2. Create New Space
- Go to: https://huggingface.co/new-space
- **Owner**: Your username (e.g., yevheniyc)
- **Space name**: `fire-safety-dashboard`
- **License**: MIT
- **SDK**: Gradio
- **Visibility**: Public
- Click **Create Space**

### 3. Push Files to Space

```bash
cd /Users/whitehat/dev/yev/pitt/huggingface_spaces/fire-safety-dashboard

# Initialize git
git init
git add .
git commit -m "Initial commit: Fire Safety Analytics Dashboard"

# Add HF remote (replace USERNAME with your HF username)
git remote add origin https://huggingface.co/spaces/USERNAME/fire-safety-dashboard

# Push
git push -u origin main
```

### 4. Wait for Build
- Hugging Face will automatically:
  - Detect `pyproject.toml` and use `uv`
  - Install dependencies
  - Launch `app.py`
  - Deploy your dashboard

### 5. Your Dashboard Will Be Live!
**URL**: `https://huggingface.co/spaces/USERNAME/fire-safety-dashboard`

---

## 📊 What Gets Deployed

✅ **Interactive Dashboard** with:
- Geographic heat maps (Folium)
- Temporal trend charts (Plotly)
- Emergency priority analysis
- False alarm cost calculator

✅ **Data**: 205K corrected fire alarm records (42MB)

✅ **Dependencies**: Managed by `uv` via `pyproject.toml`

---

## 🎯 Features

Your deployed dashboard will have:
- Real-time interactive visualizations
- Municipality comparisons
- Seasonal pattern analysis
- Policy recommendation insights
- Responsive design (works on mobile)

---

## 🔧 Troubleshooting

**If build fails:**
1. Check build logs in HF Space settings
2. Verify data file uploaded (42MB corrected_fire_alarms.csv)
3. Ensure all imports work with provided dependencies

**To update:**
```bash
# Make changes to app.py locally
git add .
git commit -m "Update: description of changes"
git push
```

Hugging Face will automatically rebuild!

---

**Your fire safety research will be live and interactive!** 🔥📊

