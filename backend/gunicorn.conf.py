# gunicorn.conf.py
# ── CHANGE 3: Use gevent async workers instead of sync workers ────────────────
# Sync workers block the entire process while reading a slow upload.
# If the upload takes longer than `timeout`, the worker is killed (WORKER TIMEOUT).
# Gevent workers are non-blocking — they handle slow clients without dying.
# Install gevent first: pip install gevent

worker_class = "gevent"
workers = 2                  # Keep low — face_recognition is CPU-heavy
worker_connections = 50      # Concurrent connections per worker
timeout = 120                # 2 minutes — enough for any normal image upload
bind = "0.0.0.0:8080"


# ── CHANGE 4: Pre-load the face database before workers fork ──────────────────
# Previously, every worker loaded the DB from Google Drive on its first request.
# After a WORKER TIMEOUT kill, the new replacement worker had to download ~3 files
# all over again (~4–5 seconds cold start per crash).
# on_starting() runs once in the master process before any workers fork.
# Workers inherit _data_cache via OS copy-on-write, so no redundant downloads.
def on_starting(server):
    from app import load_data
    load_data()
