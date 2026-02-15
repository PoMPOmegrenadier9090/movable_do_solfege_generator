from __future__ import annotations

from pathlib import Path
import uuid
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from .config import (
    ALLOWED_EXTENSIONS,
    ARTIFACTS_DIRNAME,
    CLIENTS_DIR,
    MAX_CONTENT_LENGTH,
    UPLOADS_DIRNAME,
)
from .pipeline.orchestrator import run_full_pipeline


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    CLIENTS_DIR.mkdir(parents=True, exist_ok=True)

    def is_allowed(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.post("/api/jobs")
    def create_job():
        """
        音声ファイルを受け取り，フルパイプラインを実行して結果を返すAPIエンドポイント．
        """
        if "file" not in request.files:
            return jsonify({"error": "file is required"}), 400

        uploaded = request.files["file"]
        client_id = request.form.get("client_id", "default_client").strip() or "default_client"
        filename = secure_filename(uploaded.filename or "")

        if not filename:
            return jsonify({"error": "filename is empty"}), 400
        if not is_allowed(filename):
            return jsonify({"error": f"unsupported file type: {filename}"}), 400

        client_dir = CLIENTS_DIR / client_id
        jobs_dir = client_dir / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex
        job_dir = jobs_dir / job_id
        upload_dir = job_dir / UPLOADS_DIRNAME
        artifacts_dir = job_dir / ARTIFACTS_DIRNAME
        upload_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        input_path = upload_dir / filename
        uploaded.save(input_path)

        def _noop_update(*_args, **_kwargs):
            return None

        try:
            result = run_full_pipeline(job_id, _noop_update, client_id, job_dir, input_path)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify(result), 200

    @app.get("/api/media/<client_id>/<job_id>/<path:filename>")
    def get_media(client_id: str, job_id: str, filename: str):
        media_path = CLIENTS_DIR / client_id / "jobs" / job_id / ARTIFACTS_DIRNAME / filename
        if not media_path.exists() or not media_path.is_file():
            return jsonify({"error": "media not found"}), 404
        return send_file(Path(media_path))

    return app
