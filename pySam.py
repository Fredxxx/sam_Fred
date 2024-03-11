"""
pySAM: this is an easy implementation of SAM via python to produce an executable file
it is based on https://github.com/facebookresearch/segment-anything?ref=blog.roboflow.com
and follows https://blog.roboflow.com/how-to-use-segment-anything-model-sam/
"""
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import sys
from PyQt5 import QtWidgets


class MainWindow(QtWidgets.QMainWindow):
    # define a few default variable for easy handling of live mode and more
    imgLoadPath = "NA"              # path to the image to load
    mask_predictor = []             # object of SAM
    img = []                        # original image, high res
    imgAnno = []                    # annotated image with resize factor
    nameWin = "image to annotate"   # name of display window
    resizeFactor = 2                # resize factor of displayed image
    h = 2048                        # height of image
    w = 2048                        # width of image
    masks = []                      # masks returned form sam
    scores = []                     # scores of sam prediction

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("pySAM")
        self.setGeometry(0, 0, 300, 350)

        # annotate button to start magic
        self.imgSelectButton = QtWidgets.QPushButton('start annotation', self)
        self.imgSelectButton.clicked.connect(self.selectImgFile)
        self.imgSelectButton.move(20, 30)
        self.imgSelectButton.resize(200, 30)

        # check image button, to see if image is ok to annotate (size, intensity, ...)
        self.startButton = QtWidgets.QPushButton('Check image', self)
        self.startButton.clicked.connect(self.checkImg)
        self.startButton.move(140, 100)
        self.startButton.resize(100, 30)

        # scaling factor to fit image to screen
        self.resizeNumLabel = QtWidgets.QLabel("scaling factor:", self)
        self.resizeNumLabel.move(20, 100)
        self.resizeNumLabel.resize(100, 30)

        # scaling factor input
        self.resizeNumLE = QtWidgets.QLineEdit(self)
        self.resizeNumLE.setText("2")
        self.resizeNumLE.move(90, 105)
        self.resizeNumLE.resize(20, 20)

        # brief manual to display
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

        self.loadModel()            # loads at start

    def checkImg(self):
        # displays an image to check if suitable to annotate
        cv2.destroyAllWindows()     # destroys if a window is still open
        self.loadImg()              # loads image
        flag = True                 # helper flag for display loop
        while flag:                 # display loop
            cv2.imshow("preview", self.imgAnno)     # show image in preview window
            k = cv2.waitKey(20) & 0xFF              # wait signal for ....
            if k == ord('a'):                       # .... a press
                cv2.destroyWindow("preview")        # closes preview window
                flag = False
            elif cv2.getWindowProperty("preview", cv2.WND_PROP_VISIBLE) < 1:    # ... window close
                flag = False

    def loadImg(self):
        # loads an image with openCV
        imgLoadPathTub = QtWidgets.QFileDialog.getOpenFileName(self, "Select image to annotate") # select path of image
        if imgLoadPathTub[0]:  # if a path was selected -> display image
            self.imgLoadPath = imgLoadPathTub[0]        # path as string
            print("pySAM: loading image")
            self.img = cv2.imread(self.imgLoadPath)     # load image
            print("pySAM: image loaded")
            self.resizeFactor = int(self.resizeNumLE.text())    # get resize factor
            self.h, self.w = self.img.shape[:2]                 # get image pixel number in height and width
            self.h = int(self.h / self.resizeFactor)            # change number of pixels in height
            self.w = int(self.w / self.resizeFactor)            # change number of pixels in width
            self.imgAnno = cv2.resize(self.img, (self.w, self.h)) # resize image
            print("pySAM: image resized")
        else:  # if no path was selected
            print("pySAM ERROR: no img load image string found. (loadImg)")

    def selectImgFile(self):
        self.loadImg()
        self.startSAM()

    def loadModel(self):
        # loads sam model
        print("pySam: select model")
        modelLoadPath = QtWidgets.QFileDialog.getOpenFileName(self, "Select SAM model")  # model path selector
        if modelLoadPath[0]:
            dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # check if GPU is available
            modelType = "vit_b"  # needs model name which needs to be included in path
            if "vit_b" in modelLoadPath[0]:
                modelType = "vit_b"
            elif "vit_l" in modelLoadPath[0]:
                modelType = "vit_l"
            elif "vit_h" in modelLoadPath[0]:
                modelType = "vit_h"
            else:
                print("pySAM ERROR: could not match model name to type string (LoadModel)")
            print("pySam: loading model ", modelType)
            sam = sam_model_registry[modelType](checkpoint=modelLoadPath[0]).to(device=dev)  # sam registry
            self.mask_predictor = SamPredictor(sam)     # mask predictor
            print("pySam: model loaded")
        else:
            print("pySAM ERROR: no model load image string found. (loadModel)")
            self.loadModel()  # try again to load model

    def startSAM(self):
        # magic: starts annotation
        print("pySAM: setting image to mask predictor (this might take some time)")
        # some stuff needed
        # sets image to mask predictor (I am not sure what that is doing, but it is time-consuming)
        self.mask_predictor.set_image(self.img)
        print("pySAM: set image to mask predictor finished!")
        cv2.namedWindow(self.nameWin, cv2.WINDOW_FULLSCREEN)    # name display window for easy access
        imgMask = np.uint8(np.zeros(self.img.shape))            # null array for saving

        # mouse event listener
        def onMouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:  # left button event -> select (add) mask
                input_point = np.array([[x*self.resizeFactor, y*self.resizeFactor]])  # [X,Y]-coordinates of mouse cursor
                # predict mask for specific point
                self.masks, self.scores, logits = self.mask_predictor.predict(
                    point_coords=input_point,
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                # add current mask to all overview
                imgMask[self.masks[np.argmax(self.scores)], 2] = 255
                self.c = 0  # a mark for reverting (see RBUTTONDOWN)
                print('x = %d, y = %d  added with score = %0.2f ' % (x, y, np.max(self.scores)))
            elif event == cv2.EVENT_MBUTTONDOWN:  # middle button event, similar to left button event -> subtract mask
                input_point = np.array([[x * self.resizeFactor, y * self.resizeFactor]])
                self.masks, self.scores, logits = self.mask_predictor.predict(
                    point_coords=input_point,
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
                imgMask[self.masks[np.argmax(self.scores)], 2] = 0
                self.c = 1
                print('x = %d, y = %d  subtract with score = %0.2f ' % (x, y, np.max(self.scores)))
            elif event == cv2.EVENT_RBUTTONDOWN:  # middle button event, revert changes
                if self.c == 0:  # revert last selected mask
                    imgMask[self.masks[np.argmax(self.scores)], 2] = 0
                elif self.c == 1:  # revert last subtracted mask
                    imgMask[self.masks[np.argmax(self.scores)], 2] = 255
                print('reverted')
        cv2.setMouseCallback(self.nameWin, onMouse)  # set mouse event listener to window

        # display window
        flag = True  # interruption flag
        while flag:
            imgMaskAnno = cv2.resize(imgMask, (self.w, self.h))  # annotation mask, for display only
            cv2.imshow(self.nameWin, cv2.addWeighted(self.imgAnno, 1, imgMaskAnno, 0.2, 0))  # show overlay of image and annotation mask
            k = cv2.waitKey(20) & 0xFF    # wait for keyboad signal
            if k == ord('s'):  # save image and exits annotation pipeline
                cv2.destroyWindow(self.nameWin)                     # closes image
                splitS = self.imgLoadPath.split(".")                # prepare saving str
                imgSavePath1 = splitS[0] + "_mask." + splitS[1]     # prepare saving str
                cv2.imwrite(imgSavePath1, imgMask)                  # save image mask
                print("pySAM: image saved, annotation stopped")
                flag = False                                        # stop display loop
            elif k == ord('c'):  # save image and continue annotation pipeline with next image
                cv2.destroyWindow(self.nameWin)                     # closes image
                splitS = self.imgLoadPath.split(".")                # prepare saving str
                imgSavePath1 = splitS[0] + "_mask." + splitS[1]     # prepare saving str
                cv2.imwrite(imgSavePath1, imgMask)                  # save image mask
                print("pySAM: image saved, continue annotation with loading new image")
                self.loadImg()                                      # load new image
                self.startSAM()                                     # start new annotation pipeline
                flag = False                                        # stop display loop
            elif k == ord('a'):  # aborts annotation
                cv2.destroyWindow(self.nameWin)                     # stop display loop
                flag = False                                        # stop display loop
            elif cv2.getWindowProperty(self.nameWin, cv2.WND_PROP_VISIBLE) < 1:  # aborts annotation when clicking "X" on window (upper right)
                flag = False                                        # stop display loop


def main():
    app = QtWidgets.QApplication([])
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
