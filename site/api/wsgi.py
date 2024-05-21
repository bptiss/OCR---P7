from backend import app
import os
import socket

hostname = socket.getfqdn()
print("IP Address:",socket.gethostbyname_ex(hostname)[2][1])
local_ip = socket.gethostbyname_ex(hostname)[2][0]
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host=local_ip, port=5555)
