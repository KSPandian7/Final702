import cv2
import matplotlib.pyplot as plt

sketch = cv2.imread(r"E:\19AI702-P1\data\sketches\person_010.jpg", 0)
photo  = cv2.imread(r"E:\19AI702-P1\data\photos\person_010.jpg", 0)

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(sketch, cmap='gray')
plt.title("Forensic Sketch")

plt.subplot(1,2,2)
plt.imshow(photo, cmap='gray')
plt.title("Mugshot Photo")

plt.axis('off')
plt.show()
