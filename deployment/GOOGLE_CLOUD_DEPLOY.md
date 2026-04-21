# Google Cloud Deployment (Cloud Run)

This project is ready to deploy on Google Cloud Run.

## 1) Prerequisites

- Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
- Have a Google Cloud project with billing enabled
- Enable APIs:
  - Cloud Run API
  - Cloud Build API
  - Artifact Registry API

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## 2) Build and Deploy

Run from project root:

```bash
gcloud run deploy hcpgnn-auditor ^
  --source . ^
  --region us-central1 ^
  --allow-unauthenticated ^
  --port 8000 ^
  --memory 2Gi ^
  --cpu 2
```

PowerShell users can use backtick continuation instead of `^`.

## 3) Verify

After deployment, Google Cloud prints a service URL.

Check health:

```bash
curl https://YOUR_SERVICE_URL/health
```

Run detection:

```bash
curl -X POST https://YOUR_SERVICE_URL/api/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"source_code\":\"pragma solidity ^0.8.0; contract A { function x() public {} }\"}"
```

## 4) Share for Review Demo

- Open `https://YOUR_SERVICE_URL/app` to show the visual interface.
- Keep one sample vulnerable contract and one safe contract ready.
- Show API docs at `https://YOUR_SERVICE_URL/docs`.
