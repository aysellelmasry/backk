import os, io, json, pickle, requests, numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
from PIL import Image, ImageOps
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class Config:
    GDRIVE_ENCODINGS_ID = os.getenv('GDRIVE_ENCODINGS_ID', '1cuvndmsrehLX6uZK8C30U1AlQ82HY2oG')
    GDRIVE_METADATA_ID  = os.getenv('GDRIVE_METADATA_ID',  '17O01aMqPGO0xO5A8G7qjOSrQv3ZxMu08')
    GDRIVE_MAPPING_ID   = os.getenv('GDRIVE_MAPPING_ID',   '17WUEwVKK5oydc6VRV4HJXV1KZ0q0cYGf')
    TOLERANCE           = float(os.getenv('TOLERANCE', '0.52'))
    MAX_UPLOAD_MB       = int(os.getenv('MAX_UPLOAD_MB', '16'))
    GDRIVE_DOWNLOAD     = "https://drive.google.com/uc?export=download&id={}&confirm=t"
    GDRIVE_DIRECT       = "https://drive.google.com/uc?export=view&id={}"
    GDRIVE_THUMB        = "https://drive.google.com/thumbnail?id={}&sz=w500"


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_UPLOAD_MB * 1024 * 1024

# ── CORS: allow ALL origins (public read-only API, no auth needed) ────────────
CORS(app, resources={r"/*": {"origins": "*"}},
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"])

_data_cache = None


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
    for chunk in r.iter_content(chunk_size=8192):
        buf.write(chunk)
    return buf.getvalue()


def load_data():
    global _data_cache
    if _data_cache is not None:
        return _data_cache

    logger.info("Loading face database from Google Drive...")

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

    enc_array = (np.array(enc_matrix, dtype=np.float64)
                 if enc_matrix else np.empty((0, 128), dtype=np.float64))
    logger.info(f"Encoding matrix: {enc_array.shape}")

    _data_cache = (db, meta, gdrive, ids, enc_array)
    return _data_cache


def encode_uploaded_images(files):
    encodings = []
    for file in files:
        if not file or file.filename == '':
            continue
        try:
            img = Image.open(file.stream)
            img = ImageOps.exif_transpose(img).convert('RGB')
            img.thumbnail((1200, 1200), Image.LANCZOS)
            arr = np.array(img)
            found = face_recognition.face_encodings(arr, num_jitters=3, model='large')
            if found:
                encodings.append(found[0])
            else:
                logger.warning(f"No face in {file.filename}")
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
    return encodings


def get_gdrive_urls(filename: str, gdrive_map: dict):
    file_id = gdrive_map.get(filename) or gdrive_map.get(os.path.splitext(filename)[0])
    if not file_id:
        return None, None
    return Config.GDRIVE_DIRECT.format(file_id), Config.GDRIVE_THUMB.format(file_id)


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Graduation Face Search API", "version": "1.0"})


@app.route('/health', methods=['GET'])
def health():
    db, meta, gdrive, ids, enc_array = load_data()
    return jsonify({
        "status": "healthy",
        "total_photos": len(meta),
        "total_faces": len(enc_array),
        "gdrive_mapped": len(gdrive),
    })


@app.route('/search-face', methods=['POST', 'OPTIONS'])
def search_face():
    if request.method == 'OPTIONS':
        return '', 204

    files = request.files.getlist('face_image')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No images uploaded. Field name must be "face_image".'}), 400

    query_encodings = encode_uploaded_images(files)
    if not query_encodings:
        return jsonify({'error': 'No face detected. Use a clear, well-lit, front-facing photo.'}), 400

    query_enc = np.mean(query_encodings, axis=0)
    db, meta, gdrive, ids, enc_array = load_data()

    if len(enc_array) == 0:
        return jsonify({'success': True, 'matches': [], 'total_found': 0,
                        'warning': 'Face database is empty.'})

    distances = face_recognition.face_distance(enc_array, query_enc)

    best = {}
    for photo_id, dist in zip(ids, distances):
        if photo_id not in best or dist < best[photo_id]:
            best[photo_id] = dist

    matches, skipped = [], 0
    for photo_id, dist in best.items():
        if dist >= Config.TOLERANCE:
            continue
        info = meta.get(photo_id, {})
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
    return jsonify({
        'success':           True,
        'matches':           matches,
        'total_found':       len(matches),
        'skipped_no_gdrive': skipped,
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Max {Config.MAX_UPLOAD_MB} MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error.', 'detail': str(e)}), 500


if __name__ == '__main__':
    load_data()
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
