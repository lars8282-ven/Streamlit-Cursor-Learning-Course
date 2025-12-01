# Deploying to Streamlit Cloud

This guide will walk you through deploying your Oil & Gas Analytics Dashboard to Streamlit Cloud.

## Prerequisites

1. A GitHub account (free)
2. Your code pushed to a GitHub repository
3. A Streamlit Cloud account (free at https://share.streamlit.io)

## Step-by-Step Deployment

### Step 1: Create a GitHub Repository

1. Go to https://github.com and sign in
2. Click the "+" icon in the top right → "New repository"
3. Name your repository (e.g., `oil-gas-analytics`)
4. Choose **Public** (required for free Streamlit Cloud)
5. **DO NOT** initialize with README, .gitignore, or license (if you already have files)
6. Click "Create repository"

### Step 2: Push Your Code to GitHub

If you haven't initialized git yet, run these commands in your project directory:

```bash
git init
git add .
git commit -m "Initial commit: Oil & Gas Analytics Dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

**Important files to include:**
- `app.py` (your main Streamlit app)
- `requirements.txt` (dependencies)
- `README_STREAMLIT.md` (optional but recommended)

### Step 3: Sign Up for Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click "Sign up" or "Get started"
3. Sign in with your GitHub account
4. Authorize Streamlit Cloud to access your GitHub repositories

### Step 4: Deploy Your App

1. In Streamlit Cloud, click "New app"
2. Select your repository from the dropdown
3. Select the branch (usually `main` or `master`)
4. **Main file path:** Enter `app.py` (since your app is in the root)
5. Click "Deploy"

### Step 5: Wait for Deployment

- Streamlit Cloud will automatically:
  - Install dependencies from `requirements.txt`
  - Run your app
  - Provide you with a public URL

The deployment usually takes 1-3 minutes.

## Configuration Options

### Custom App URL

After deployment, you can customize your app URL:
- Go to your app settings in Streamlit Cloud
- Click "Settings" → "General"
- Change the app subdomain (if available)

### Environment Variables (if needed)

If you need to add environment variables:
1. Go to your app in Streamlit Cloud
2. Click "Settings" → "Secrets"
3. Add your secrets in TOML format

## Troubleshooting

### Common Issues

1. **App won't deploy:**
   - Check that `requirements.txt` is in the root directory
   - Ensure `app.py` is in the root or update the main file path
   - Check the deployment logs for errors

2. **Dependencies fail to install:**
   - Verify all packages in `requirements.txt` are available on PyPI
   - Check for version conflicts
   - Some packages may not be available for all platforms

3. **App crashes on load:**
   - Check the logs in Streamlit Cloud dashboard
   - Ensure all imports are in `requirements.txt`
   - Verify your code doesn't rely on local files that aren't in the repo

### Viewing Logs

1. Go to your app in Streamlit Cloud
2. Click "Manage app" → "Logs"
3. Review error messages and stack traces

## Updating Your App

To update your deployed app:

1. Make changes to your code locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update app"
   git push
   ```
3. Streamlit Cloud will automatically detect the changes and redeploy

## Best Practices

1. **Keep requirements.txt updated:** Always include all dependencies
2. **Use relative paths:** Don't use absolute file paths
3. **Test locally first:** Make sure your app works with `streamlit run app.py`
4. **Optimize data loading:** Use `@st.cache_data` for expensive operations
5. **Keep secrets secure:** Never commit API keys or passwords to GitHub

## Your App Structure

Your current project structure is perfect for Streamlit Cloud:
```
your-repo/
├── app.py              ← Main Streamlit app (will be detected automatically)
├── requirements.txt    ← Dependencies (required)
└── README_STREAMLIT.md ← Documentation (optional)
```

## Next Steps

After deployment, you'll get a URL like:
`https://your-app-name.streamlit.app`

Share this URL with anyone who needs access to your dashboard!

