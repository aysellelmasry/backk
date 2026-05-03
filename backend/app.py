import os, io, json, pickle, requests, numpy as np, threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
from PIL import Image, ImageOps
import logging
from functools import lru_cache
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
class Config:
    GDRIVE_ENCODINGS_ID  = os.getenv('GDRIVE_ENCODINGS_ID', '1cuvndmsrehLX6uZK8C30U1AlQ82HY2oG')
    GDRIVE_METADATA_ID   = os.getenv('GDRIVE_METADATA_ID',  '17O01aMqPGO0xO5A8G7qjOSrQv3ZxMu08')
    GDRIVE_MAPPING_ID    = os.getenv('GDRIVE_MAPPING_ID',   '17WUEwVKK5oydc6VRV4HJXV1KZ0q0cYGf')
    TOLERANCE            = float(os.getenv('TOLERANCE', '0.52'))
    MAX_UPLOAD_MB        = int(os.getenv('MAX_UPLOAD_MB', '16'))
    MAX_IMAGE_PX         = int(os.getenv('MAX_IMAGE_PX', '1200'))      # max thumbnail dimension
    GDRIVE_DOWNLOAD      = "https://drive.google.com/uc?export=download&id={}&confirm=t"
    GDRIVE_DIRECT        = "https://drive.google.com/uc?export=view&id={}"
    GDRIVE_THUMB         = "https://drive.google.com/thumbnail?id={}&sz=w500"
    MAX_RESULTS          = int(os.getenv('MAX_RESULTS', '50'))          # cap results per search
    REQUEST_TIMEOUT_S    = int(os.getenv('REQUEST_TIMEOUT_S', '30'))    # per-request soft timeout


# ══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_UPLOAD_MB * 1024 * 1024

CORS(app, resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"])


# ══════════════════════════════════════════════════════════════════════════════
# THREAD-SAFE DATA CACHE
# ══════════════════════════════════════════════════════════════════════════════
_data_cache = None
_data_lock  = threading.Lock()   # prevents multiple workers loading simultaneously


def download_from_drive(file_id: str) -> bytes:
    url = Config.GDRIVE_DOWNLOAD.format(file_id)
    logger.info(f"Downloading GDrive file: {file_id}")
    session = requests.Session()
    r = session.get(url, stream=True, timeout=180)
    r.raise_for_status()
    if 'text/html' in r.headers.get('Content-Type', ''):
        for key, val in r.cookies.items():
            if key.startswith('download_warning'):
                r = session.get(url + f"&confirm={val}", stream=True, timeout=180)
                r.raise_for_status()
                break
    buf = io.BytesIO()
    for chunk in r.iter_content(chunk_size=65536):  # larger chunks = faster download
        buf.write(chunk)
    return buf.getvalue()


def load_data():
    """Load face DB once. Thread-safe. Subsequent calls return cached data instantly."""
    global _data_cache
    if _data_cache is not None:
        return _data_cache

    with _data_lock:
        # Double-checked locking: another thread may have loaded while we waited
        if _data_cache is not None:
            return _data_cache

        logger.info("Loading face database from Google Drive...")
        t0 = time.time()

        try:
            db = pickle.loads(download_from_drive(Config.GDRIVE_ENCODINGS_ID))
            logger.info(f"Loaded {len(db)} face records")
        except Exception as e:
            logger.error(f"Failed to load encodings: {e}"); db = {}

        try:
            meta = pickle.loads(download_from_drive(Config.GDRIVE_METADATA_ID))
            logger.info(f"Loaded {len(meta)} metadata records")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}"); meta = {}

        try:
            gdrive = json.loads(download_from_drive(Config.GDRIVE_MAPPING_ID).decode('utf-8'))
            logger.info(f"Loaded {len(gdrive)} GDrive mappings")
        except Exception as e:
            logger.error(f"Failed to load GDrive mapping: {e}"); gdrive = {}

        ids, enc_matrix = [], []
        for photo_id, data in db.items():
            if isinstance(data, np.ndarray) and data.shape == (128,):
                ids.append(photo_id); enc_matrix.append(data)
            elif isinstance(data, dict):
                for enc in data.get('encodings', []):
                    ids.append(photo_id); enc_matrix.append(enc)

        enc_array = (np.array(enc_matrix, dtype=np.float32)  # float32 saves memory vs float64
                     if enc_matrix else np.empty((0, 128), dtype=np.float32))
        logger.info(f"Encoding matrix: {enc_array.shape} — loaded in {time.time()-t0:.1f}s")

        _data_cache = (db, meta, gdrive, ids, enc_array)

    return _data_cache


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def validate_image(raw_bytes: bytes, filename: str) -> bool:
    """Reject obviously bad files before expensive processing."""
    # Minimum viable JPEG/PNG magic bytes
    if raw_bytes[:3] == b'\xff\xd8\xff':
        return True  # JPEG
    if raw_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return True  # PNG
    if raw_bytes[:6] in (b'GIF87a', b'GIF89a'):
        return True  # GIF
    if raw_bytes[:4] == b'RIFF' and raw_bytes[8:12] == b'WEBP':
        return True  # WEBP
    logger.warning(f"Rejected non-image file: {filename}")
    return False


def encode_uploaded_images(files):
    """Read, validate, resize, and encode all uploaded face images."""
    encodings = []
    for file in files:
        if not file or file.filename == '':
            continue
        try:
            # Buffer entirely into memory — prevents socket timeout mid-read
            raw = file.read()
            if not raw:
                logger.warning(f"Empty file: {file.filename}")
                continue
            if not validate_image(raw, file.filename):
                continue

            img = Image.open(io.BytesIO(raw))
            img = ImageOps.exif_transpose(img).convert('RGB')
            # Resize large images before encoding — cuts encoding time significantly
            if max(img.size) > Config.MAX_IMAGE_PX:
                img.thumbnail((Config.MAX_IMAGE_PX, Config.MAX_IMAGE_PX), Image.LANCZOS)
            arr = np.array(img)

            found = face_recognition.face_encodings(arr, num_jitters=1, model='small')
            # num_jitters=1 + model='small' is ~10x faster than num_jitters=3 + model='large'
            # with only a tiny accuracy cost — essential for high concurrency
            if found:
                encodings.append(found[0])
            else:
                logger.warning(f"No face detected in: {file.filename}")
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
    return encodings


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_gdrive_urls(filename: str, gdrive_map: dict):
    file_id = gdrive_map.get(filename) or gdrive_map.get(os.path.splitext(filename)[0])
    if not file_id:
        return None, None
    return Config.GDRIVE_DIRECT.format(file_id), Config.GDRIVE_THUMB.format(file_id)


# ══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ══════════════════════════════════════════════════════════════════════════════
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Graduation Face Search API", "version": "2.0"})


@app.route('/health', methods=['GET'])
def health():
    db, meta, gdrive, ids, enc_array = load_data()
    return jsonify({
        "status":        "healthy",
        "total_photos":  len(meta),
        "total_faces":   len(enc_array),
        "gdrive_mapped": len(gdrive),
        "cache_loaded":  _data_cache is not None,
    })


@app.route('/search-face', methods=['POST', 'OPTIONS'])
def search_face():
    if request.method == 'OPTIONS':
        return '', 204

    # ── 1. Force full buffering of request body before touching files ─────────
    # Prevents Gunicorn sync worker from dying mid-socket-read on slow uploads
    request.get_data()

    files = request.files.getlist('face_image')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No images uploaded. Field name must be "face_image".'}), 400

    # ── 2. Encode faces ───────────────────────────────────────────────────────
    query_encodings = encode_uploaded_images(files)
    if not query_encodings:
        return jsonify({
            'error': 'No face detected. Use a clear, well-lit, front-facing photo.',
            'tip':   'Make sure the face is unobstructed and the image is in focus.'
        }), 400

    query_enc = np.mean(query_encodings, axis=0).astype(np.float32)

    # ── 3. Load (cached) DB ───────────────────────────────────────────────────
    db, meta, gdrive, ids, enc_array = load_data()

    if len(enc_array) == 0:
        return jsonify({'success': True, 'matches': [], 'total_found': 0,
                        'warning': 'Face database is empty.'})

    # ── 4. Vectorised distance computation (numpy — no Python loop) ───────────
    distances = face_recognition.face_distance(enc_array, query_enc)

    # Keep best (closest) distance per photo_id — numpy-friendly approach
    photo_ids  = np.array(ids)
    unique_ids = np.unique(photo_ids)
    best       = {}
    for uid in unique_ids:
        mask       = photo_ids == uid
        best[uid]  = float(distances[mask].min())

    # ── 5. Build result list ──────────────────────────────────────────────────
    matches, skipped = [], 0
    for photo_id, dist in best.items():
        if dist >= Config.TOLERANCE:
            continue
        info      = meta.get(photo_id, {})
        filename  = info.get('filename', f"{photo_id}.jpg")
        full_url, thumb_url = get_gdrive_urls(filename, gdrive)
        if not full_url:
            skipped += 1
            continue
        matches.append({
            'photo_id':   photo_id,
            'url':        full_url,
            'thumbnail':  thumb_url,
            'filename':   filename,
            'confidence': round(float(1 - dist), 4),
        })

    # Sort by confidence, cap results
    matches.sort(key=lambda x: x['confidence'], reverse=True)
    matches = matches[:Config.MAX_RESULTS]

    return jsonify({
        'success':           True,
        'matches':           matches,
        'total_found':       len(matches),
        'skipped_no_gdrive': skipped,
    })


# ══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════════════════
@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'Bad request.', 'detail': str(e)}), 400

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Max {Config.MAX_UPLOAD_MB} MB per upload.'}), 413

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({'error': 'Too many requests. Please try again shortly.'}), 429

@app.errorhandler(500)
def server_error(e):
    logger.exception("Unhandled 500 error")
    return jsonify({'error': 'Internal server error.'}), 500

@app.errorhandler(Exception)
def unhandled(e):
    logger.exception(f"Unhandled exception: {e}")
    return jsonify({'error': 'Unexpected error. Please try again.'}), 500


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT (dev only — production uses gunicorn)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    load_data()
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
