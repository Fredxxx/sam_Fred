import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import sys
from PyQt5 import QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    imgLoadPath = "NA"
    mask_predictor = []
    img = []
    imgAnno = []
    nameWin = "image to annotate"
    resizeFactor = 2
    h = 2048
    w = 2048
    masks = []
    scores = []

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("pySAM")
        self.setGeometry(0, 0, 300, 350)

        self.imgSelectButton = QtWidgets.QPushButton('start annotation', self)
        self.imgSelectButton.clicked.connect(self.selectImgFile)
        self.imgSelectButton.move(20, 30)
        self.imgSelectButton.resize(200, 30)

        self.startButton = QtWidgets.QPushButton('Check image', self)
        self.startButton.clicked.connect(self.checkImg)
        self.startButton.move(140, 100)
        self.startButton.resize(100, 30)

        self.resizeNumLabel = QtWidgets.QLabel("scaling factor:", self)
        self.resizeNumLabel.move(20, 100)
        self.resizeNumLabel.resize(100, 30)

        self.resizeNumLE = QtWidgets.QLineEdit(self)
        self.resizeNumLE.setText("2")
        self.resizeNumLE.move(90, 105)
        self.resizeNumLE.resize(20, 20)

        self.tipsLabel = QtWidgets.QLabel("Manual: \n"
                                          "I.   \" start annotation\": \n"
                                          "         1. select image to annotate \n"
                                          "         2. mark structures of interest \n"
                                          "             a. \"left click\" annotate structure \n"
                                          "             b. \"right click\" revert last action (slightly dodgy)\n"
                                          "             c. \"middle click\" subtract structure (poor performance, would need more work) \n"
                                          "         3. save mask by pressing: \n"
                                          "             a. save and exit \"s\" \n"
                                          "             b. save and continue with next image \"c\" \n"
                                          "II.  \"Check image\": to check if image display is good enough for annotation \n"
                                          "     (image needs to be 8bit with good intensity scaling, also important for network) \n"
                                          "III.  \"scaling factor\": to resize image to match screen (display purpose only) \n"
                                          "IV.  press \"a\" to abort operation (only if img is selected)\n\n"
                                          "\"Disclaimer\": \n"
                                          "This was written in a hurry so there is room for improvements and bugs all over the place!\n"
                                          "frederik.goerlitz@mail.medizin.uni-freiburg.de"
                                          , self)
        self.tipsLabel.setWordWrap(True)
        self.tipsLabel.move(20, 150)
        self.tipsLabel.resize(500, 240)

        self.loadModel()

        #self.alphaLabel = QtWidgets.QLabel("contrast:", self)
        #self.alphaLabel.move(20, 120)
        #self.alphaLabel.resize(100, 30)

        #self.alphaLE = QtWidgets.QLineEdit(self)
        #self.alphaLE.setText("2")
        #self.alphaLE.move(120, 125)
        #self.alphaLE.resize(20, 20)

        #self.betaLabel = QtWidgets.QLabel("brightness:", self)
        #self.betaLabel.move(20, 140)
        #self.betaLabel.resize(100, 30)

        #self.betaLE = QtWidgets.QLineEdit(self)
        #self.betaLE.setText("20")
        #self.betaLE.move(120, 145)
        #self.betaLE.resize(20, 20)

    def checkImg(self):
        cv2.destroyAllWindows()
        self.loadImg()
        flag = True
        while flag:
            cv2.imshow("preview", self.imgAnno)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('a'):
                cv2.destroyWindow("preview")
                flag = False
            elif cv2.getWindowProperty("preview", cv2.WND_PROP_VISIBLE) < 1:
                flag = False

    def loadImg(self):
        imgLoadPathTub = QtWidgets.QFileDialog.getOpenFileName(self, "Select image to annotate")
        if imgLoadPathTub[0]:
            self.imgLoadPath = imgLoadPathTub[0]
            print("pySAM: loading image")
            self.img = cv2.imread(self.imgLoadPath)
            print("pySAM: image loaded")
            self.resizeFactor = int(self.resizeNumLE.text())
            self.h, self.w = self.img.shape[:2]
            self.h = int(self.h / self.resizeFactor)
            self.w = int(self.w / self.resizeFactor)
            self.imgAnno = cv2.resize(self.img, (self.w, self.h))
            print("pySAM: image resized")
        else:
            print("pySAM ERROR: no img load image string found. (loadImg)")

    def selectImgFile(self):
        self.loadImg()
        self.startSAM()

    def loadModel(self):
        print("pySam: select model")
        modelLoadPath = QtWidgets.QFileDialog.getOpenFileName(self, "Select SAM model")
        if modelLoadPath[0]:
            dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            modelType = "vit_b"
            if "vit_b" in modelLoadPath[0]:
                modelType = "vit_b"
            elif "vit_l" in modelLoadPath[0]:
                modelType = "vit_l"
            elif "vit_h" in modelLoadPath[0]:
                modelType = "vit_h"
            else:
                print("pySAM ERROR: could not match model name to type string (LoadModel)")
            print("pySam: loading model ", modelType)
            sam = sam_model_registry[modelType](checkpoint=modelLoadPath[0]).to(device=dev)
            self.mask_predictor = SamPredictor(sam)
            print("pySam: model loaded")
        else:
            print("pySAM ERROR: no model load image string found. (loadModel)")
            self.loadModel()

    def startSAM(self):
        print("pySAM: setting image to mask predictor (this might take some time)")
        self.mask_predictor.set_image(self.img)
        print("pySAM: set image to mask predictor finished!")
        # Provide points as input prompt [X,Y]-coordinates

        cv2.namedWindow(self.nameWin, cv2.WINDOW_FULLSCREEN)
        imgMask = np.uint8(np.zeros(self.img.shape))

        def onMouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                input_point = np.array([[x*self.resizeFactor, y*self.resizeFactor]])
                self.masks, self.scores, logits = self.mask_predictor.predict(
                    point_coords=input_point,
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                imgMask[self.masks[np.argmax(self.scores)], 2] = 255
                self.c = 0
                print('x = %d, y = %d  added with score = %0.2f ' % (x, y, np.max(self.scores)))
            elif event == cv2.EVENT_MBUTTONDOWN:
                input_point = np.array([[x * self.resizeFactor, y * self.resizeFactor]])
                self.masks, self.scores, logits = self.mask_predictor.predict(
                    point_coords=input_point,
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                imgMask[self.masks[np.argmax(self.scores)], 2] = 0
                self.c = 1
                print('x = %d, y = %d  subtract with score = %0.2f ' % (x, y, np.max(self.scores)))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.c == 0:
                    imgMask[self.masks[np.argmax(self.scores)], 2] = 0
                elif self.c == 1:
                    imgMask[self.masks[np.argmax(self.scores)], 2] = 255
                print('reverted')

        cv2.setMouseCallback(self.nameWin, onMouse)

        flag = True
        while flag:
            imgMaskAnno = cv2.resize(imgMask, (self.w, self.h))
            cv2.imshow(self.nameWin, cv2.addWeighted(self.imgAnno, 1, imgMaskAnno, 0.2, 0))
            k = cv2.waitKey(20) & 0xFF
            if k == ord('s'):
                cv2.destroyWindow(self.nameWin)
                splitS = self.imgLoadPath.split(".")
                imgSavePath1 = splitS[0] + "_mask." + splitS[1]
                cv2.imwrite(imgSavePath1, imgMask)
                print("pySAM: stopped")
                flag = False
            elif k == ord('c'):
                cv2.destroyWindow(self.nameWin)
                splitS = self.imgLoadPath.split(".")
                imgSavePath1 = splitS[0] + "_mask." + splitS[1]
                cv2.imwrite(imgSavePath1, imgMask)
                self.loadImg()
                self.startSAM()
                flag = False
            elif k == ord('a'):
                cv2.destroyWindow(self.nameWin)
                flag = False
            elif cv2.getWindowProperty(self.nameWin, cv2.WND_PROP_VISIBLE) < 1:
                flag = False


def main():
    app = QtWidgets.QApplication([])
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
