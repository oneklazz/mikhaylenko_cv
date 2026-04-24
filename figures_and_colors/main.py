import cv2

img = cv2.imread("data/balls_and_rects.png")

sv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue = sv[:, :, 0]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t, bi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
alf, n = cv2.findContours(bi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rect = {}
circ = {}

for c in alf:
    x, y, w, h = cv2.boundingRect(c)
    cx = x + w // 2
    cy = y + h // 2
    color = hue[cy, cx]

    if len(c) > 4:
        if color not in circ:
            circ[color] = 0
        circ[color] += 1
    else:
        if color not in rect:
            rect[color] = 0
        rect[color] += 1

print("Всего фигур:", len(alf))

print("Прямоугольники:")
for i, k in enumerate(sorted(rect), 1):
    print(i, "Оттенок:", rect[k])

print("Круги:")
for i, k in enumerate(sorted(circ), 1):
    print(i, "Оттенок:", circ[k])
