import cv2
import os

dir = "gt_train"
names = []
outcome = []
def convert(contours):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x_center = (x + w / 2) / len1
        y_center = (y + h / 2) / len0
        w_normalized = w / len1
        h_normalized = h / len0
        output = "0 " + str(x_center) + " " + str(y_center) + " " + str(w_normalized) + " " + str(h_normalized)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    names.append(filename)
    outcome.append(output)
    return names, outcome

for filename in os.listdir(r"./"+dir):
    #print(filename)
    img = cv2.imread(dir + "/" + filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = img.shape
    len1 = size[1]
    len0 = size[0]
    #print(size)
    retval, new = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contour, hierarchy = cv2.findContours(new, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    names,outcome = convert(contour)

for i in names:
    i = i.strip('_segmentation.png')

print(names)
print(outcome)