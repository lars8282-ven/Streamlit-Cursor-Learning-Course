# Quick Deploy to Streamlit Cloud

## Fast Track (5 minutes)

### 1. Push to GitHub
```bash
git init
git add app.py requirements.txt README_STREAMLIT.md
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repo
5. Main file: `app.py`
6. Click "Deploy"

Done! Your app will be live at `https://your-app-name.streamlit.app`

## Requirements Checklist
- ✅ `app.py` in root directory
- ✅ `requirements.txt` with all dependencies
- ✅ Repository is **Public** (for free tier)
- ✅ Code pushed to GitHub

