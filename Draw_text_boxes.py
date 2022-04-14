import cv2
from CRAFT_TextDetector import detect_text
from Deep_TextRecognition import text_recog

graph_img = cv2.imread('/Users/daggubatisirichandana/PycharmProjects/chart_percept/Improve_test/limit/2/2.png')

img = graph_img.copy()
detected_label,boxes,box_centers,slope = detect_text(img)
boxes = boxes.astype(int)
for i in boxes:
    img = cv2.rectangle(img, tuple(i[0]), tuple(i[2]), (0, 0, 255), 2)
cv2.imwrite("/Users/daggubatisirichandana/PycharmProjects/chart_percept/Improve_test/limit/2/2_text.png",img)
