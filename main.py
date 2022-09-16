import torch
import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("./param/images/IMG_3310.PNG")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./param/best.pt')
results = model(image)
print(results)
fig, ax = plt.subplots(figsize=(16, 12))
ax.imshow(results.render()[0])
plt.show()