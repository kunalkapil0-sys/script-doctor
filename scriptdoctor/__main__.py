"""Command line:  python -m scriptdoctor script.txt --title "..." --format short --market india [--json]
Local page+API: python -m scriptdoctor --serve 8000   then open http://127.0.0.1:8000 , or POST /analyze {"script": "...", "title": "...", "format": "short", "market": "india"}
"""
import argparse, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from .analyze import ScriptDoctor

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")


def text_report(r):
    out = [f"\nSCRIPT DOCTOR  {r['score']}/100  {r['verdict']}",
           f"craft {r['craft_score']}/100" + (f" | packaging percentile {r['packaging']['percentile']} (model tested at {r['packaging']['tested_accuracy']:.0%})" if r["packaging"] else " | no title given"),
           f"~{r['runtime_seconds']} s, {r['spoken_words']} spoken words, est. average viewed {r['avg_retention_estimate']}%", "",
           "Scores: " + ", ".join(f"{k} {v}/10" for k, v in r["scores"].items()), "", "Retention estimate:"]
    for i, b in enumerate(r["retention"], 1):
        bar = "#" * round(b["est"] / 5)
        out.append(f" {i:>2}. {b['pos']:>3}%  {bar:<20} {b['est']:>3}%  {b['risk']:<4} {b['label']}" + (f"  <- {', '.join(b['reasons'])}" if b["reasons"] else ""))
    out += ["", "Fix first:"]
    for f in r["fixes"]:
        out.append(f" [P{f['priority']}] {f['issue']}")
        if f.get("quote"): out.append(f"       \"{f['quote']}\"")
        out.append(f"       {f['suggestion']}")
        if f.get("example"): out.append(f"       Try opening with: \"{f['example']}\"")
    out += ["", "Keep: " + " | ".join(r["keep"]), "Ask yourself: " + " | ".join(r["questions"]), "", r["honesty"]["note"]]
    return "\n".join(out)


def serve(port, sd):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_OPTIONS(self): self._send(204, {})

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = open(PAGE, "rb").read()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            elif self.path == "/health":
                self._send(200, {"ok": True, "metrics": sd.metrics})
            else:
                self._send(404, {"error": "open / for the page, POST /analyze for the API"})

        def do_POST(self):
            if self.path != "/analyze": return self._send(404, {"error": "POST /analyze"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 200_000: return self._send(413, {"error": "script too long"})
                d = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, sd.analyze(d.get("script", ""), d.get("title", ""), d.get("format", "short"), d.get("market", "india"), d.get("tags", "")))
            except ValueError as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *a): pass

    print(f"Script Doctor running: open http://127.0.0.1:{port} in your browser  (API: POST /analyze, GET /health)")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser(prog="scriptdoctor", description="Score a video script: craft, retention estimate, title, fixes.")
    ap.add_argument("file", nargs="?", help="script .txt file (or - for stdin)")
    ap.add_argument("--title", default=""); ap.add_argument("--tags", default="")
    ap.add_argument("--format", default="short", choices=["short", "long"]); ap.add_argument("--market", default="india", choices=["india", "global"])
    ap.add_argument("--json", action="store_true"); ap.add_argument("--serve", type=int, metavar="PORT")
    a = ap.parse_args(); sd = ScriptDoctor()
    if a.serve: return serve(a.serve, sd)
    if not a.file: ap.error("give a script file, or --serve PORT")
    text = sys.stdin.read() if a.file == "-" else open(a.file, encoding="utf-8").read()
    r = sd.analyze(text, a.title, a.format, a.market, a.tags)
    print(json.dumps(r, ensure_ascii=False, indent=1) if a.json else text_report(r))


if __name__ == "__main__":
    main()
