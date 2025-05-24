#!/usr/bin/env python3
"""
Simple HTTP Server for Photo Collection Website
Usage: python server.py [port]
Default port: 8000
"""

import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

# Default port
PORT = 8000

# Get port from command line argument if provided
if len(sys.argv) > 1:
    try:
        PORT = int(sys.argv[1])
    except ValueError:
        print("Invalid port number. Using default port 8000.")

# Set the current directory as the web root
web_dir = Path(__file__).parent
print(f"Serving files from: {web_dir.absolute()}")

# Change to the web directory
import os
os.chdir(web_dir)

# Create the server
Handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(("", PORT), Handler)

print(f"Photo Collection Website running at:")
print(f"Local: http://localhost:{PORT}")
print(f"Network: http://0.0.0.0:{PORT}")
print(f"\nPress Ctrl+C to stop the server")

# Try to open the browser automatically
try:
    webbrowser.open(f'http://localhost:{PORT}')
    print(f"Opening browser...")
except:
    print(f"Could not open browser automatically. Please visit http://localhost:{PORT}")

# Start the server
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print(f"\nServer stopped.")
    httpd.shutdown() 