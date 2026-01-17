from http.server import SimpleHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import os
import threading
import signal
import sys
import socket
import qrcode
import ssl
from pathlib import Path

PORT = 8000

# -----------------------------
# Shutdown event
# -----------------------------
shutdown_event = threading.Event()

# -----------------------------
# Ctrl+C handler with confirmation
# -----------------------------
def ctrl_c_handler(signum, frame):
    print("\nCtrl+C detected.")
    while True:
        answer = input("Do you really want to shut down the server? (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            print("Shutting down server...")
            shutdown_event.set()
            server.shutdown()  # stops serve_forever()
            break
        elif answer in ("n", "no"):
            print("Continuing server...")
            break
        else:
            print("Please answer 'y' or 'n'.")

signal.signal(signal.SIGINT, ctrl_c_handler)

# -----------------------------
# Discover local IPv4 addresses
# -----------------------------
def get_local_ipv4_addresses():
    addresses = set()
    hostname = socket.gethostname()
    for info in socket.getaddrinfo(hostname, None):
        family, _, _, _, sockaddr = info
        if family == socket.AF_INET:
            ip = sockaddr[0]
            if not ip.startswith("127."):
                addresses.add(ip)
    addresses.add("127.0.0.1")
    return sorted(addresses)

# -----------------------------
# Change working directory
# -----------------------------
web_dir = os.path.join(os.path.dirname(__file__), "..", "Web_Page")
os.chdir(web_dir)

# -----------------------------
# Ensure QR_Codes folder exists
# -----------------------------
qr_folder = os.path.join(os.path.dirname(__file__), "QR_Codes")
os.makedirs(qr_folder, exist_ok=True)

# -----------------------------
# Path to your certificates
# -----------------------------
secure_folder = os.path.join(os.path.dirname(__file__), "..", "Server", "Secure")
cert_file = os.path.join(secure_folder, "cert.pem")
key_file = os.path.join(secure_folder, "key.pem")

if not Path(cert_file).exists() or not Path(key_file).exists():
    print(f"Error: Certificates not found in {secure_folder}")
    print("Make sure cert.pem and key.pem exist.")
    sys.exit(1)

# -----------------------------
# Threaded HTTP Server
# -----------------------------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True  # threads exit when server stops

server = ThreadedHTTPServer(("0.0.0.0", PORT), SimpleHTTPRequestHandler)

# -----------------------------
# Wrap the server socket with SSL (Python 3.10+)
# -----------------------------
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain(certfile=cert_file, keyfile=key_file)
server.socket = context.wrap_socket(server.socket, server_side=True)

# -----------------------------
# Generate QR codes
# -----------------------------
print("\nServer is reachable at (HTTPS):")
for ip in get_local_ipv4_addresses():
    url = f"https://{ip}:{PORT}/"
    print(f"  → {url}")

    qr = qrcode.QRCode(version=1, box_size=10, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    safe_ip = ip.replace(".", "_").replace(":", "_")
    filename = os.path.join(qr_folder, f"{safe_ip}.png")
    img.save(filename)
    print(f"     QR code saved -> {filename}")

print("\nPress Ctrl+C to stop the server.\n")

# -----------------------------
# Run server in a thread
# -----------------------------
def run_server():
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass  # already handled

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# -----------------------------
# Wait until shutdown_event
# -----------------------------
shutdown_event.wait()

# -----------------------------
# Cleanup
# -----------------------------
print("Cleaning up server...")
server.server_close()
sys.exit(0)
