# Endoskin Deployment Guide

## Requirements
- Python 3.9.x
- MongoDB Atlas account (for database)

## Deployment to Render.com

1. **Create a new Web Service** on Render.com
2. **Connect your GitHub/GitLab repository**
3. **Configure environment variables**:
   - `MONGODB_URI`: Your MongoDB connection string
4. **Set build settings**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
5. **Deploy**

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| MONGODB_URI | Yes | MongoDB connection string |
| PYTHON_VERSION | No | Python version (default: 3.9.13) |

## Local Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables:
   ```bash
   export MONGODB_URI="your_mongodb_uri"
   ```

4. Run the application:
   ```bash
   python app.py
   ```
