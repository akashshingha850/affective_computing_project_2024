import cv2
import numpy as np

img = np.zeros((300, 600, 3), dtype=np.uint8)  # Create a blank image
cv2.putText(img, 'Hello, World!', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow('Test Window', img)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()
