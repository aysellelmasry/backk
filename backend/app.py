import os
import io
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
from PIL import Image, ImageOps
import logging

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
class Config:
    # Paths — resolve relative to this file so the app works from any CWD
    BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR            = os.getenv('DATA_DIR', BASE_DIR)
    ENCODINGS_FILE      = os.path.join(DATA_DIR, os.getenv('ENCODINGS_FILE',  'face_encodings.pkl'))
    METADATA_FILE       = os.path.join(DATA_DIR, os.getenv('METADATA_FILE',   'photos_metadata.pkl'))
    GDRIVE_MAPPING_FILE = os.path.join(DATA_DIR, os.getenv('GDRIVE_FILE',     'gdrive_file_mapping.json'))

    TOLERANCE      = float(os.getenv('TOLERANCE',    '0.52'))
    MAX_UPLOAD_MB  = int(os.getenv('MAX_UPLOAD_MB',  '16'))
    PORT           = int(os.getenv('PORT',           '5000'))

    # Allowed origins — comma-separated list or * for all
    # Example: ALLOWED_ORIGINS=https://myapp.vercel.app,https://myapp2.netlify.app
    ALLOWED_ORIGINS = [o.strip() for o in os.getenv('ALLOWED_ORIGINS', '*').split(',') if o.strip()]

    GDRIVE_DIRECT = "https://drive.google.com/uc?export=view&id={}"
    GDRIVE_THUMB  = "https://drive.google.com/thumbnail?id={}&sz=w500"

# ── App setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_UPLOAD_MB * 1024 * 1024

origins = Config.ALLOWED_ORIGINS if Config.ALLOWED_ORIGINS != ['*'] else '*'
CORS(app,
     resources={r"/*": {"origins": origins}},
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "OPTIONS"],
     supports_credentials=False)
logger.info(f"CORS origins: {origins}")

# ── Data cache ────────────────────────────────────────────────────────────
_data_cache = None

def load_data():
    global _data_cache
    if _data_cache is not None:
        return _data_cache

    logger.info("Loading face database…")

    # Face encodings
    try:
        with open(Config.ENCODINGS_FILE, 'rb') as f:
            db = pickle.load(f)
        logger.info(f"  {len(db)} encoding records loaded")
    except FileNotFoundError:
        logger.error(f"MISSING: {Config.ENCODINGS_FILE}")
        db = {}
    except Exception as e:
        logger.error(f"Encodings load error: {e}")
        db = {}

    # Photo metadata
    try:
        with open(Config.METADATA_FILE, 'rb') as f:
            meta = pickle.load(f)
        logger.info(f"  {len(meta)} metadata records loaded")
    except FileNotFoundError:
        logger.error(f"MISSING: {Config.METADATA_FILE}")
        meta = {}
    except Exception as e:
        logger.error(f"Metadata load error: {e}")
        meta = {}

    # Google Drive file-ID mapping
    try:
        with open(Config.GDRIVE_MAPPING_FILE, 'r') as f:
            gdrive = json.load(f)
        logger.info(f"  {len(gdrive)} GDrive mappings loaded")
    except FileNotFoundError:
        logger.error(f"MISSING: {Config.GDRIVE_MAPPING_FILE}")
        gdrive = {}
    except Exception as e:
        logger.error(f"GDrive map load error: {e}")
        gdrive = {}

    # Build encoding matrix
    ids, enc_list = [], []
    for photo_id, data in db.items():
        if isinstance(data, np.ndarray) and data.shape == (128,):
            # Flat single encoding
            ids.append(photo_id)
            enc_list.append(data)
        else:
            for enc in data.get('encodings', []):
                ids.append(photo_id)
                enc_list.append(enc)

    if enc_list:
        enc_array = np.array(enc_list, dtype=np.float64)
        logger.info(f"  Encoding matrix: {enc_array.shape}")
    else:
        enc_array = np.empty((0, 128), dtype=np.float64)
        logger.warning("  Encoding matrix is EMPTY — run your indexing script first!")

    _data_cache = (db, meta, gdrive, ids, enc_array)
    return _data_cache

# ── Helpers ───────────────────────────────────────────────────────────────
def encode_uploaded_images(files):
    encodings = []
    for file in files:
        if not file or file.filename == '':
            continue
        try:
            img = Image.open(file.stream)
            img = ImageOps.exif_transpose(img)
            img = img.convert('RGB')
            img.thumbnail((1200, 1200), Image.LANCZOS)
            arr = np.array(img)
            found = face_recognition.face_encodings(arr, num_jitters=3, model='large')
            if found:
                encodings.append(found[0])
                logger.info(f"  Face encoded from {file.filename}")
            else:
                logger.warning(f"  No face in {file.filename}")
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
    return encodings

def get_gdrive_urls(filename, gdrive_map):
    file_id = gdrive_map.get(filename)
    if not file_id:
        file_id = gdrive_map.get(os.path.splitext(filename)[0])
    if not file_id:
        return None, None
    return (
        Config.GDRIVE_DIRECT.format(file_id),
        Config.GDRIVE_THUMB.format(file_id),
    )

# ── CORS preflight catch-all ──────────────────────────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# ── Routes ────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "Graduation Photo Search API", "version": "1.0"})

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    db, meta, gdrive, ids, enc_array = load_data()
    missing = [f for f in [Config.ENCODINGS_FILE, Config.METADATA_FILE, Config.GDRIVE_MAPPING_FILE]
               if not os.path.exists(f)]
    return jsonify({
        "status":        "healthy",
        "total_photos":  len(meta),
        "total_faces":   len(enc_array),
        "gdrive_mapped": len(gdrive),
        "tolerance":     Config.TOLERANCE,
        "missing_files": missing,
        "db_records":    len(db),
    })

@app.route('/search-face', methods=['POST', 'OPTIONS'])
def search_face():
    if request.method == 'OPTIONS':
        return '', 204

    files = request.files.getlist('face_image')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No images uploaded. Use field name "face_image".'}), 400

    query_encodings = encode_uploaded_images(files)
    if not query_encodings:
        return jsonify({
            'error': (
                'No face detected in your uploaded photo(s). '
                'Please use a clear, well-lit, front-facing photo.'
            )
        }), 400

    query_enc = np.mean(query_encodings, axis=0)

    db, meta, gdrive, ids, enc_array = load_data()
    if len(enc_array) == 0:
        return jsonify({
            'success': True, 'matches': [], 'total_found': 0,
            'warning': 'Face database is empty. Run your indexing script first.'
        })

    distances = face_recognition.face_distance(enc_array, query_enc)
    logger.info(f"Distances — min:{distances.min():.3f} max:{distances.max():.3f} "
                f"hits:{(distances < Config.TOLERANCE).sum()}")

    # Best distance per photo
    best = {}
    for photo_id, dist in zip(ids, distances):
        if photo_id not in best or dist < best[photo_id]:
            best[photo_id] = dist

    matches, skipped = [], 0
    for photo_id, dist in best.items():
        if dist >= Config.TOLERANCE:
            continue
        info     = meta.get(photo_id, {})
        filename = info.get('filename', f"{photo_id}.jpg")
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

    matches.sort(key=lambda x: x['confidence'], reverse=True)
    logger.info(f"Done: {len(matches)} matches, {skipped} skipped (no GDrive)")

    return jsonify({
        'success':           True,
        'matches':           matches,
        'total_found':       len(matches),
        'skipped_no_gdrive': skipped,
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Max {Config.MAX_UPLOAD_MB} MB per upload.'}), 413

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500: {e}")
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

if __name__ == '__main__':
    load_data()
    app.run(host='0.0.0.0', port=Config.PORT, debug=os.getenv('FLASK_DEBUG', '').lower() == 'true')
