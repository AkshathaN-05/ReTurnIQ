"""
Vercel Serverless Function Entrypoint for ReTurnIQ FastAPI Backend.
Exposes the FastAPI `app` instance for Vercel Python runtime deployment.
"""

from api.main import app
