#!/usr/bin/env bash
set -euo pipefail
# In later steps we'll prepare models, then start the server.
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
