# gunicorn.conf.py
# Production configuration for high concurrency (hundreds of users)

import multiprocessing

# ── Workers ───────────────────────────────────────────────────────────────────
# gevent = async, non-blocking workers. Essential for handling slow uploads
# without killing the worker on every request.
# Install: pip install gevent
worker_class       = "gevent"
worker_connections = 200          # concurrent connections per worker

# face_recognition is CPU-bound. Too many workers = CPU thrashing.
# Formula: 2 × CPU cores is the standard starting point.
# On a 2-core machine: 4 workers. On a 4-core: 8 workers.
# Override with env var GUNICORN_WORKERS if needed.
import os
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2))

# ── Timeouts ──────────────────────────────────────────────────────────────────
timeout        = 120    # worker killed if no response in 120s (was 30s — too short)
graceful_timeout = 30   # how long to wait for in-flight requests on shutdown
keepalive      = 5      # seconds to keep idle connections open

# ── Binding ───────────────────────────────────────────────────────────────────
bind = f"0.0.0.0:{os.getenv('PORT', '8080')}"

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog  = "-"    # stdout
errorlog   = "-"    # stderr
loglevel   = "info"
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sµs'

# ── Performance ───────────────────────────────────────────────────────────────
# Recycle workers after N requests to prevent memory leaks from face_recognition
max_requests          = 500
max_requests_jitter   = 50   # randomise so all workers don't restart at once


# ── Pre-load face DB before workers fork ─────────────────────────────────────
# This is the most important production setting.
# Without this, every worker cold-starts by downloading ~3 files from Google Drive
# on its first request. With hundreds of users, new workers spin up constantly.
# on_starting() runs ONCE in the master process. Workers inherit the loaded cache
# via OS copy-on-write fork — zero redundant downloads ever.
def on_starting(server):
    server.log.info("Pre-loading face database before workers fork...")
    from app import load_data
    load_data()
    server.log.info("Face database pre-loaded. Workers will inherit cache.")


# ── Worker lifecycle hooks ────────────────────────────────────────────────────
def post_fork(server, worker):
    server.log.info(f"Worker spawned (pid: {worker.pid})")

def worker_exit(server, worker):
    server.log.info(f"Worker exited (pid: {worker.pid})")
