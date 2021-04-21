import os
import glob
import cv2
import matplotlib.pyplot as plt

root = 'E:\\ai_learning_resource\\hwdb\\HWDB1\\train\\'
# w_max = 0
# h_max = 0
for clazz in sorted(os.listdir(os.path.join(root))[6]):
    for x in glob.glob(os.path.join(str(root), str(clazz), '*.png'))[:1]:
        # img = Image.open(x).convert('L')
        # print(img.mode, img.format)
        origineImage = cv2.imread(x)
        image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
        retval, img = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
        plt.imshow(img)
        plt.show()
