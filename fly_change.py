import cv2
import numpy as np
from matplotlib import pyplot as plt

# 傅里叶变换
# f = np.fft.fft2(img)
# fshift = np.fft.fft2(f)
# magnitude_spectrum = 20 * np.log(np.abs(fshift))
#
# rows, cols = img.shape
# crow, ccol = int(rows / 2), int(cols / 2)
# fshift[crow - 20: crow + 20, ccol - 20: ccol + 20] = 0
# f_shift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_shift)
# img_back = np.abs(img_back)
#
# plt.subplot(221), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]),plt.yticks([])
# plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.subplot(223),plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
# plt.subplot(224),plt.imshow(img_back, cmap = 'gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])

# plt.subplot(121), plt.imshow(img, 'gray'), plt.title('origin')
# plt.xticks([]), plt.yticks([])
#
# rows, cols = img.shape
# mask = np.ones(img.shape, np.uint8)
# mask[int(rows / 2) - 50: int(rows / 2) + 50, int(cols / 2) - 50: int(cols / 2) - 50] = 1
# f1 = np.fft.fft2(img)
# f1shift = np.fft.fftshift(f1)
# f1shift = f1shift * mask
# f2shift = np.fft.ifftshift(f1shift)
# img_new = np.fft.ifft2(f2shift)
# img_new = np.abs(img_new)
# img_new = (img_new - np.amin(img_new) / (np.amax(img_new) - np.amin(img_new)))
# plt.subplot(122), plt.imshow(img_new, 'gray'), plt.title('highpass')
# plt.xticks([]), plt.yticks([])

def fly_calculate(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    fshift[crow - 20: crow + 20, ccol - 20: ccol + 20] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum2 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 20: crow + 20, ccol - 20: ccol + 20] = 1

    f_dft_shift = dft_shift * mask
    f_idft_shift = np.fft.ifftshift(f_dft_shift)

    img_back2 = cv2.idft(f_idft_shift)
    img_back2 = cv2.magnitude(img_back2[:, :, 0], img_back2[:, :, 1])

    imgList = [img, magnitude_spectrum, img_back, img, magnitude_spectrum2, img_back2]
    imgName = ['img', 'magnitude_spectrum', 'img_back', 'img', 'magnitude_spectrum2', 'img_back2']

    for i in range(6):
        # plt.subplot(2, 3, i + 1), \
        plt.imshow(imgList[i], cmap='gray')
            # , plt.title(imgName[i])
        # plt.xticks([]), plt.yticks([])
    plt.show()


img = cv2.imread('images/2.jpeg', 0)
fly_calculate(img)
img = cv2.imread('images/0.jpeg', 0)
fly_calculate(img)
img = cv2.imread('images/3.jpeg', 0)
fly_calculate(img)
