import numpy as np
import matplotlib.pyplot as plt
import socket

host = "84.237.21.36"
port = 5152

def get_data(sock, n):
    data = bytearray()
    while len(data) < n:
        part = sock.recv(n - len(data))
        if not part:
            return None
        data.extend(part)
    return data

plt.ion()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((host, port))
    s.send(b"124ras1")
    print(s.recv(10))

    ans = b"nope"

    while ans != b"yep":
        s.send(b"get")
        data = get_data(s, 40002)
        h = data[0]
        w = data[1]
        img = np.frombuffer(data[2:], dtype="uint8")

        i1 = np.argmax(img)
        p1 = np.unravel_index(i1, (h, w))
        y, x = p1
        y1 = max(0, y - 10)
        y2 = min(h - 1, y + 10)
        x1 = max(0, x - 10)
        x2 = min(w - 1, x + 10)

        for yy in range(y1, y2 + 1):
            for xx in range(x1, x2 + 1):
                ind = yy * w + xx
                img[ind] = 0

        i2 = np.argmax(img)
        p2 = np.unravel_index(i2, (h, w))
        d = ((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)**0.5
        d = round(d, 1)
        s.send(f"{d}".encode())
        print(s.recv(10))

        s.send(b"beat")
        ans = s.recv(10)
