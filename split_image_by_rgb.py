from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# img = Image.open('images/4.jpg')
img = np.zeros((200, 200, 3), np.uint8)
img[:] = (255, 0 , 0)
# gray = img.convert('L')
# gray = np.array(gray)
# plt.figure('rgb')
# plt.imshow(gray, cmap='gray')
# plt.axis('on')

r, g, b = img.split()
pic = Image.merge('RGB', (r, g, b))

plt.figure("RGB")
plt.subplot(2,3,1), plt.title('origin')
plt.imshow(img), plt.axis('on')

plt.subplot(2,3,2), plt.title('gray')
plt.imshow(gray, cmap='gray'), plt.axis('off')

plt.subplot(2,3,3), plt.title('merge')   # merge 合并
plt.imshow(pic),plt.axis('off')
# 红、绿、蓝三个通道的缩略图，都是以灰度显示的，用不同的灰度色阶来表示“ 红，绿，蓝”在图像中的比重。
r = np.array(r)
plt.subplot(2,3,4), plt.title('r')
plt.imshow(r),plt.axis('off')

g = np.array(g)
plt.subplot(2,3,5), plt.title('g')
plt.imshow(g),plt.axis('off')

b = np.array(b)
plt.subplot(2,3,6), plt.title('b')
plt.imshow(b),plt.axis('off')

plt.show()

