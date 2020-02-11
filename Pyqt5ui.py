from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi
from butterworth import Butter
import sys
import cv2
import numpy as np
import scipy.signal as sig
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

class LoadQt(QMainWindow):
    def __init__(self):
        super(LoadQt, self).__init__()
        loadUi('demo.ui', self)
        self.setWindowIcon(QtGui.QIcon("python-icon.png"))

        self.image = None
        self.actionOpen.triggered.connect(self.open_img)
        self.actionSave.triggered.connect(self.save_img)
        self.actionPrint.triggered.connect(self.createPrintDialog)
        self.actionQuit.triggered.connect(self.QuestionMessage)
        self.actionBig.triggered.connect(self.big_Img)
        self.actionSmall.triggered.connect(self.small_Img)
        self.actionQt.triggered.connect(self.AboutMessage)
        self.actionAuthor.triggered.connect(self.AboutMessage2)

        # Chương 2
        self.actionRotation.triggered.connect(self.rotation)
        self.actionAffine.triggered.connect(self.shearing)
        self.actionTranslation.triggered.connect(self.translation)

        # Chương 3
        self.actioAnhXam.triggered.connect(self.anh_Xam)
        self.actionNegative.triggered.connect(self.anh_Negative)
        self.actionHistogram.triggered.connect(self.histogram_Equalization)
        self.actionLog.triggered.connect(self.Log)
        self.actionGamma.triggered.connect(self.gamma)

        # Image Restoration
        self.actionGaussian.triggered.connect(self.gaussian_noise)
        self.actionRayleigh.triggered.connect(self.rayleigh_noise)
        self.actionErlang.triggered.connect(self.erlang_noise)
        self.actionUniform.triggered.connect(self.uniform_noise)
        self.actionImpluse.triggered.connect(self.impulse_noise)
        self.actionHistogram_PDF.triggered.connect(self.hist)

        # Image Restoration 1
        self.actionAdaptive_Wiener_Filtering.triggered.connect(self.weiner_filter)
        self.actionMedian_Filtering.triggered.connect(self.median_filtering)
        self.actionAdaptive_Median_Filtering.triggered.connect(self.adaptive_median_filtering)

        # Image Restoration 2
        self.actionInverse_Filter.triggered.connect(self.inv_filter)

        # Simple Edge Detection
        self.actionSHT.triggered.connect(self.simple_edge_detection)

        # Smoothing
        self.actionBlur.triggered.connect(self.blur)
        self.actionBox_Filter.triggered.connect(self.box_filter)
        self.actionMedian_Filter.triggered.connect(self.median_filter)
        self.actionBilateral_Filter.triggered.connect(self.bilateral_filter)
        self.actionGaussian_Filter.triggered.connect(self.gaussian_filter)

        # Filter
        self.actionMedian_threshold_2.triggered.connect(self.median_threshold)
        self.actionDirectional_Filtering_2.triggered.connect(self.directional_filtering)
        self.actionDirectional_Filtering_3.triggered.connect(self.directional_filtering2)
        self.actionDirectional_Filtering_4.triggered.connect(self.directional_filtering3)
        self.action_Butterworth.triggered.connect(self.butter_filter)
        self.action_Notch_filter.triggered.connect(self.notch_filter)

        # Cartooning of an Image
        self.actionCartoon.triggered.connect(self.cartoon)

        # Set input
        self.dial.valueChanged.connect(self.rotation2)
        self.horizontalSlider.valueChanged.connect(self.Gamma_)
        self.gaussian_QSlider.valueChanged.connect(self.gaussian_filter2)
        self.erosion.valueChanged.connect(self.erode)
        self.Qlog.valueChanged.connect(self.Log)
        self.size_Img.valueChanged.connect(self.SIZE)
        self.canny.stateChanged.connect(self.Canny)
        self.canny_min.valueChanged.connect(self.Canny)
        self.canny_max.valueChanged.connect(self.Canny)
        self.pushButton.clicked.connect(self.reset)

    @pyqtSlot()
    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.tmp = self.image
        self.displayImage()

    def displayImage(self, window=1):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        # image.shape[0] là số pixel theo chiều Y
        # image.shape[1] là số pixel theo chiều X
        # image.shape[2] lưu số channel biểu thị mỗi pixel
        img = img.rgbSwapped() # chuyển đổi hiệu quả một ảnh RGB thành một ảnh BGR.
        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)# căn chỉnh vị trí xuất hiện của hình trên lable
        if window == 2:
            self.imgLabel2.setPixmap(QPixmap.fromImage(img))
            self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def open_img(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\Users\DELL\PycharmProjects\DemoPro', "Image Files (*)")
        if fname:
            self.loadImage(fname)
        else:
            print("Invalid Image")

    def save_img(self):
        fname, filter = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\', "Image Files (*.png)")
        if fname:
            cv2.imwrite(fname, self.image) # Lưu trữ ảnh
            print("Error")

    def createPrintDialog(self):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)

        if dialog.exec_() == QPrintDialog.Accepted:
            self.imgLabel2.print_(printer)

    def big_Img(self):
        self.image = cv2.resize(self.image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def small_Img(self):
        self.image = cv2.resize(self.image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def SIZE(self , c):
        self.image = self.tmp
        self.image = cv2.resize(self.image, None, fx=c, fy=c, interpolation=cv2.INTER_CUBIC)
        self.displayImage(2)

    def reset(self):
        self.image = self.tmp
        self.displayImage(2)

    def AboutMessage(self):
        QMessageBox.about(self, "About Qt - Qt Designer",
            "Qt is a multiplatform C + + GUI toolkit created and maintained byTrolltech.It provides application developers with all the functionality needed to build applications with state-of-the-art graphical user interfaces.\n"
            "Qt is fully object-oriented, easily extensible, and allows true component programming.Read the Whitepaper for a comprehensive technical overview.\n\n"

            "Since its commercial introduction in early 1996, Qt has formed the basis of many thousands of successful applications worldwide.Qt is also the basis of the popular KDE Linux desktop environment, a standard component of all major Linux distributions.See our Customer Success Stories for some examples of commercial Qt development.\n\n"

            "Qt is supported on the following platforms:\n\n"

                "\tMS / Windows - - 95, 98, NT\n"
                "\t4.0, ME, 2000, and XP\n"
                "\tUnix / X11 - - Linux, Sun\n"
                "\tSolaris, HP - UX, Compaq Tru64 UNIX, IBM AIX, SGI IRIX and a wide range of others\n"
                "\tMacintosh - - Mac OS X\n"
                "\tEmbedded - - Linux platforms with framebuffer support.\n\n"
                          
            "Qt is released in different editions:\n\n"
            
                "\tThe Qt Enterprise Edition and the Qt Professional Edition provide for commercial software development.They permit traditional commercial software distribution and include free upgrades and Technical Support.For the latest prices, see the Trolltech web site, Pricing and Availability page, or contact sales @ trolltech.com.The Enterprise Edition offers additional modules compared to the Professional Edition.\n\n"
                "\tThe Qt Open Source Edition is available for Unix / X11, Macintosh and Embedded Linux.The Open Source Edition is for the development of Free and Open Source software only.It is provided free of charge under the terms of both the Q Public License and the GNU General Public License."
        )
    def AboutMessage2(self):
        QMessageBox.about(self, "About Author", "Người hướng dẫn:   NGÔ QUỐC VIỆT \n\n" 
                                                "Người thực hiện:\n" 
                                                    "\tPhan Hoàng Việt - 42.01.104.189"
                          )

    def QuestionMessage(self):
        message = QMessageBox.question(self, "Exit", "Bạn có chắc muốn thoát", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if message == QMessageBox.Yes:
            print("Yes")
            self.close()
        else:
            print("No")

################################ Chương 2 ##############################################################################
    def rotation(self):
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1) #thay đổi chiều của ảnh
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)

    def rotation2(self, angle):
        self.image = self.tmp
        rows, cols, steps = self.image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))
        self.displayImage(2)

    def shearing(self):
        self.image = self.tmp
        rows, cols, ch = self.image.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

        M = cv2.getAffineTransform(pts1, pts2)
        self.image = cv2.warpAffine(self.image, M, (cols, rows))

        self.displayImage(2)

    def translation(self):
        self.image = self.tmp
        num_rows, num_cols = self.image.shape[:2]

        translation_matrix = np.float32([[1, 0, 70], [0, 1, 110]])
        img_translation = cv2.warpAffine(self.image, translation_matrix, (num_cols, num_rows))
        self.image = img_translation
        self.displayImage(2)

    def erode(self , iter):
        self.image = self.tmp
        if iter > 0 :
            kernel = np.ones((4, 7), np.uint8)
            self.image = cv2.erode(self.tmp, kernel, iterations=iter)
        else :
            kernel = np.ones((2, 6), np.uint8)
            self.image = cv2.dilate(self.image, kernel, iterations=iter*-1)
        self.displayImage(2)

    def Canny(self):
        self.image = self.tmp
        if self.canny.isChecked():
            can = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.Canny(can, self.canny_min.value(), self.canny_max.value())
        self.displayImage(2)

################################ Chương 3 ##############################################################################
    def anh_Xam(self):
        self.image = self.tmp
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.displayImage(2)

    def anh_Xam2(self):
        self.image = self.tmp
        if self.gray.isChecked():
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        self.displayImage(2)

    def anh_Negative(self):
        self.image = self.tmp
        self.image = ~self.image
        self.displayImage(2)

    def histogram_Equalization(self):
        self.image = self.tmp
        img_yuv = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        self.image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        self.displayImage(2)

    def Log(self):
        self.image = self.tmp
        img_2 = np.uint8(np.log(self.image))
        c = 2
        self.image = cv2.threshold(img_2, c, 225, cv2.THRESH_BINARY)[1]
        self.displayImage(2)

    def Gamma_(self, gamma):
        self.image = self.tmp
        gamma = gamma*0.1
        invGamma = 1.0 /gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)

    def gamma(self):
        self.image = self.tmp
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)

#######################################Image Restoration################################################################
    def gaussian_noise(self):
        self.image = self.tmp
        row, col, ch = self.image.shape
        mean = 0
        var = 0.1
        sigma = var * 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        self.image = self.image + gauss
        self.displayImage(2)
    def erlang_noise(self):
        self.image = self.tmp
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        cv2.randu(table, 1, 1)
        self.image = cv2.LUT(self.image, table)
        self.displayImage(2)
    def rayleigh_noise(self):
        self.image = self.tmp
        r = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        self.image = cv2.randu(r, 1, 1)
        self.displayImage(2)
    def uniform_noise(self):
        self.image = self.tmp
        uniform_noise = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        cv2.randu(uniform_noise, 0, 255)
        self.image = (uniform_noise * 0.5).astype(np.uint8)
        self.displayImage(2)
    def impulse_noise(self):
        self.image = self.tmp
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(self.image)
        # Salt mode
        num_salt = np.ceil(amount * self.image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in self.image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * self.image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in self.image.shape]
        out[coords] = 0
        self.image = out
        self.displayImage(2)

    def hist(self):
        self.image = self.tmp
        histg = cv2.calcHist([self.image], [0], None, [256], [0, 256])
        self.image = histg
        plt.plot(self.image)
        plt.show()
        self.displayImage(2)

####################################Image Restoration 1#################################################################
    def median_filtering(self):
        self.image = self.tmp
        self.image = cv2.medianBlur(self.image, 5)
        self.displayImage(2)

    def adaptive_median_filtering(self):
        self.image = self.tmp
        temp = []
        filter_size = 5
        indexer = filter_size // 2
        for i in range(len(self.image)):

            for j in range(len(self.image[0])):

                for z in range(filter_size):
                    if i + z - indexer < 0 or i + z - indexer > len(self.image) - 1:
                        for c in range(filter_size):
                            temp.append(0)
                    else:
                        if j + z - indexer < 0 or j + indexer > len(self.image[0]) - 1:
                            temp.append(0)
                        else:
                            for k in range(filter_size):
                                temp.append(self.image[i + z - indexer][j + k - indexer])

                temp.sort()
                self.image[i][j] = temp[len(temp) // 2]
                temp = []
        self.displayImage(2)

    def weiner_filter(self):
        self.image = self.tmp
        M = 256  # length of Wiener filter
        Om0 = 0.1 * np.pi  # frequency of original signal
        N0 = 0.1  # PSD of additive white noise

        # generate original signal
        s = np.cos(Om0 * np.ndarray(self.image))
        # generate observed signal
        g = 1 / 20 * np.asarray([1, 2, 3, 4, 5, 4, 3, 2, 1])
        n = np.random.normal(size=self.image, scale=np.sqrt(N0))
        x = np.convolve(s, g, mode='same') + n
        # estimate (cross) PSDs using Welch technique
        f, Pxx = sig.csd(x, x, nperseg=M)
        f, Psx = sig.csd(s, x, nperseg=M)
        # compute Wiener filter
        H = Psx / Pxx
        H = H * np.exp(-1j * 2 * np.pi / len(H) * np.arange(len(H)) * (len(H) // 2))  # shift for causal filter
        h = np.fft.irfft(H)
        # apply Wiener filter to observation
        self.image = np.convolve(x, h, mode='same')
        self.displayImage(2)

####################################Image Restoration 2#################################################################
    def inv_filter(self):
        self.image = self.tmp
        for i in range(0, 3):
            g = self.image[:, :, i]
            G = (np.fft.fft2(g))

            # h = cv2.imread(self.image, 0)
            h_padded = np.zeros(g.shape)
            h_padded[:self.image.shape[0], :self.image.shape[1]] = np.copy(self.image)
            H = (np.fft.fft2(h_padded))

            # normalize to [0,1]
            H_norm = H / abs(H.max())
            G_norm = G / abs(G.max())
            F_temp = G_norm / H_norm
            F_norm = F_temp / abs(F_temp.max())

            # rescale to original scale
            F_hat = F_norm * abs(G.max())

            # 3. apply Inverse Filter and compute IFFT
            self.image = np.fft.ifft2(F_hat)
            self.image[:, :, i] = abs(self.image)
        self.displayImage(2)

##################################Simple Edge Detection#################################################################
    def simple_edge_detection(self):
        self.image = self.tmp

        # Convert the img to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection method on the image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # This returns an array of r and theta values
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        # The below for loop runs till r and theta values
        # are in the range of the 2d array
        for r, theta in lines[0]:
            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value rcos(theta)
            x0 = a * r

            # y0 stores the value rsin(theta)
            y0 = b * r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000 * (a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000 * (-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000 * (a))

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be
            # drawn. In this case, it is red.
            cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        self.displayImage(2)

#####################################Smoothing##########################################################################
    def blur(self):
        self.image = self.tmp
        self.image = cv2.blur(self.image, (5, 5))
        self.displayImage(2)
    def box_filter(self):
        self.image = self.tmp
        self.image = cv2.boxFilter(self.image, -1,(20,20))
        self.displayImage(2)
    def median_filter(self):
        self.image = self.tmp
        self.image = cv2.medianBlur(self.image,5)
        self.displayImage(2)
    def bilateral_filter(self):
        self.image = self.tmp
        self.image = cv2.bilateralFilter(self.image,9,75,75)
        self.displayImage(2)
    def gaussian_filter(self):
        self.image = self.tmp
        self.image = cv2.GaussianBlur(self.image,(5,5),0)
        self.displayImage(2)
    def gaussian_filter2(self, g):
        self.image = self.tmp
        self.image = cv2.GaussianBlur(self.image, (5, 5), g)
        self.displayImage(2)
########################################Filter##########################################################################
    def median_threshold(self):
        self.image = self.tmp
        grayscaled = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.medianBlur(self.image,5)
        retval, threshold = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.image = threshold
        self.displayImage(2)
    def directional_filtering(self):
        self.image = self.tmp
        kernel = np.ones((3, 3), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)
    def directional_filtering2(self):
        self.image = self.tmp
        kernel = np.ones((5, 5), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)
    def directional_filtering3(self):
        self.image = self.tmp
        kernel = np.ones((7, 7), np.float32) / 9
        self.image = cv2.filter2D(self.image, -1, kernel)
        self.displayImage(2)

    def butter_filter(self):
        self.image = self.tmp
        img_float32 = np.float32(self.image)

        dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.image = np.fft.fftshift(dft)

        self.image = 20 * np.log(cv2.magnitude(self.image[:, :, 0], self.image[:, :, 1]))
        self.displayImage(2)
    def notch_filter(self):
        self.image = self.tmp

        self.displayImage(2)

########################################Cartooning of an Image##########################################################
    def cartoon(self):
        num_down = 2
        num_bilateral = 7

        img_color = self.image
        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)

        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)

        img_gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)

        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY,
                                         blockSize=9,
                                         C=2)
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        self.image = cv2.bitwise_and(img_color, img_edge)

        self.displayImage(2)

########################################Moire Pattern##########################################################


app = QApplication(sys.argv)
win = LoadQt()
win.show()
sys.exit(app.exec())

