# django-image-apis

API LIVE Documentation
Deployment Status: Success (Healthy)

Platform: Koyeb (Small Instance - 1GB RAM / 1 vCPU)

1. Live API Endpoints
These endpoints are fully operational and hosted on a production-grade infrastructure to handle the memory-intensive nature of PyTorch models.

Anime Style Transfer: https://semantic-albina-zain313-7d5d6a30.koyeb.app/anime/

Mosaic Filter: https://semantic-albina-zain313-7d5d6a30.koyeb.app/mosaic/

Flower Classification: https://semantic-albina-zain313-7d5d6a30.koyeb.app/flower/

2. Technical Deployment Process
The project was moved from a local development environment to a live cloud server using the following technical steps:

Phase 1: Environment Configuration
Python Versioning: Created a .python-version file set to 3.11 to ensure the server environment matched the local development environment.

Dependency Management: Compiled a requirements.txt file including critical heavy libraries: torch, torchvision, scipy, and gunicorn.

Production Security: Updated settings.py to use dynamic ALLOWED_HOSTS via the KOYEB_PUBLIC_DOMAIN environment variable.

Phase 2: Infrastructure Setup
GitHub Integration: Connected the repository to Koyeb for automated builds and deployment.

Instance Sizing: Upgraded to a Small Instance (1GB RAM). This was necessary because the 512MB Free tier is insufficient to load 140MB PyTorch models into memory concurrently with the Django application.

Buildpack Selection: Used the automated Buildpack system to detect the Django framework and install all dependencies.

Phase 3: Runtime Execution
Static Files: Since this project consists of pure API endpoints without a frontend or user-uploaded media, WhiteNoise was intentionally omitted to keep the deployment lightweight.

Execution Command: The application is served using Gunicorn to provide a robust, production-ready interface:

gunicorn myproject.wsgi
