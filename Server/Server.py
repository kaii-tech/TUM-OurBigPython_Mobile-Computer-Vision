from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import os

PORT = 8000

# Change directory to the Web_Page folder
web_dir = os.path.join(os.path.dirname(__file__), "..", "Web_Page")
os.chdir(web_dir)

handler = SimpleHTTPRequestHandler

with TCPServer(("", PORT), handler) as httpd:
    print(f"Server running at http://localhost:{PORT}/home")
    httpd.serve_forever()
