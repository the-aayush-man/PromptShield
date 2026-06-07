from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.analyzer import analyze_prompt, model_summary
from app.reporting import create_pdf
from app.storage import clear_history, get_analysis, history, save_analysis, stats


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web"


def response_status(status: int) -> str:
    code = int(status)
    return f"{code} {HTTPStatus(code).phrase}"


def wsgi_json(start_response, payload: object, status: int = 200):
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    start_response(
        response_status(status),
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


def wsgi_static(start_response, path: str):
    if path == "/":
        path = "/index.html"

    requested = (STATIC_DIR / path.lstrip("/")).resolve()
    if STATIC_DIR.resolve() not in requested.parents and requested != STATIC_DIR:
        return wsgi_json(start_response, {"error": "Forbidden."}, HTTPStatus.FORBIDDEN)
    if not requested.exists() or not requested.is_file():
        return wsgi_json(start_response, {"error": "Not found."}, HTTPStatus.NOT_FOUND)

    body = requested.read_bytes()
    content_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
    start_response(
        response_status(HTTPStatus.OK),
        [("Content-Type", content_type), ("Content-Length", str(len(body)))],
    )
    return [body]


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")
    query = parse_qs(environ.get("QUERY_STRING", ""))

    try:
        if method == "GET" and path == "/api/health":
            return wsgi_json(start_response, {"status": "ok", "model": model_summary()})

        if method == "GET" and path == "/api/stats":
            return wsgi_json(start_response, stats())

        if method == "GET" and path == "/api/history":
            limit = int(query.get("limit", ["50"])[0])
            return wsgi_json(start_response, {"items": history(limit)})

        if method == "GET" and path.startswith("/api/report/") and path.endswith(".pdf"):
            analysis_id = path.removeprefix("/api/report/").removesuffix(".pdf")
            record = get_analysis(analysis_id)
            if record is None:
                return wsgi_json(
                    start_response,
                    {"error": "Analysis not found."},
                    HTTPStatus.NOT_FOUND,
                )
            pdf = create_pdf(record)
            start_response(
                response_status(HTTPStatus.OK),
                [
                    ("Content-Type", "application/pdf"),
                    (
                        "Content-Disposition",
                        f'attachment; filename="promptshield-report-{analysis_id}.pdf"',
                    ),
                    ("Content-Length", str(len(pdf))),
                ],
            )
            return [pdf]

        if method == "POST" and path == "/api/analyze":
            length = int(environ.get("CONTENT_LENGTH") or "0")
            raw = environ["wsgi.input"].read(length) if length else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")
            prompt = str(payload.get("prompt", "")).strip()
            result = analyze_prompt(prompt)
            analysis_id = save_analysis(prompt, result)
            result["id"] = analysis_id
            result["report_url"] = f"/api/report/{analysis_id}.pdf"
            return wsgi_json(start_response, result)

        if method == "DELETE" and path == "/api/history":
            clear_history()
            return wsgi_json(start_response, {"status": "cleared"})

        if method == "GET":
            return wsgi_static(start_response, path)

        return wsgi_json(
            start_response,
            {"error": "Unknown endpoint."},
            HTTPStatus.NOT_FOUND,
        )
    except ValueError as exc:
        return wsgi_json(start_response, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    except Exception as exc:
        return wsgi_json(
            start_response,
            {"error": f"Request failed: {exc}"},
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )


app = application


class PromptShieldHandler(BaseHTTPRequestHandler):
    server_version = "PromptShield/1.0"

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self.send_json({"status": "ok", "model": model_summary()})
            return
        if path == "/api/stats":
            self.send_json(stats())
            return
        if path == "/api/history":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["50"])[0])
            self.send_json({"items": history(limit)})
            return
        if path.startswith("/api/report/") and path.endswith(".pdf"):
            analysis_id = path.removeprefix("/api/report/").removesuffix(".pdf")
            record = get_analysis(analysis_id)
            if record is None:
                self.send_json({"error": "Analysis not found."}, HTTPStatus.NOT_FOUND)
                return
            pdf = create_pdf(record)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/pdf")
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="promptshield-report-{analysis_id}.pdf"',
            )
            self.send_header("Content-Length", str(len(pdf)))
            self.end_headers()
            self.wfile.write(pdf)
            return

        self.serve_static(path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self.send_json({"error": "Unknown endpoint."}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self.read_json()
            prompt = str(payload.get("prompt", "")).strip()
            result = analyze_prompt(prompt)
            analysis_id = save_analysis(prompt, result)
            result["id"] = analysis_id
            result["report_url"] = f"/api/report/{analysis_id}.pdf"
            self.send_json(result)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json(
                {"error": f"Analysis failed: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/history":
            clear_history()
            self.send_json({"status": "cleared"})
            return
        self.send_json({"error": "Unknown endpoint."}, HTTPStatus.NOT_FOUND)

    def serve_static(self, path: str) -> None:
        if path == "/":
            path = "/index.html"

        requested = (STATIC_DIR / path.lstrip("/")).resolve()
        if STATIC_DIR.resolve() not in requested.parents and requested != STATIC_DIR:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not requested.exists() or not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        body = requested.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PromptShield web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_summary()
    server = ThreadingHTTPServer((args.host, args.port), PromptShieldHandler)
    print(f"PromptShield running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
