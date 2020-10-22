import csv
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QGridLayout, QPushButton, QWidget
from PyQt5.QtWidgets import QFileDialog

IMAGE_SIZE = 48
X_ROWS_FILE = 'x_train_gr_smpl.csv'
IMAGES_DISPLAYED_X = 8
IMAGES_DISPLAYED_Y = 12
IMAGES_DISPLAYED = IMAGES_DISPLAYED_X * IMAGES_DISPLAYED_Y


def readCSV_Lines(min, max):
    with open(X_ROWS_FILE, "r") as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)#skip header row
        for i in range(min):
            next(datareader)
        for i in range(max-min):
            yield next(datareader)

def createImage(pixelArray):
    im = QImage(IMAGE_SIZE, IMAGE_SIZE, QImage.Format_RGB32)
    i = 0
    for pixel in pixelArray:
        pixelVal = int(float(pixel))
        im.setPixel(i % IMAGE_SIZE, i / IMAGE_SIZE, QColor(pixelVal, pixelVal, pixelVal, 255).rgb())
        i += 1
    return im

class MainWindow(QWidget):
    def leftClickCallback(self):
        if self.currentLine > 0:
            self.currentLine -= IMAGES_DISPLAYED
        gen = readCSV_Lines(self.currentLine, self.currentLine + IMAGES_DISPLAYED)
        for y in range(IMAGES_DISPLAYED_Y):
            for x in range(IMAGES_DISPLAYED_X):
                self.setDisplayedImage(self.imageLabel[x][y], next(gen))

    def rightClickCallback(self):
        if self.currentLine < 10000:
            self.currentLine += IMAGES_DISPLAYED
        gen = readCSV_Lines(self.currentLine, self.currentLine + IMAGES_DISPLAYED)
        for y in range(IMAGES_DISPLAYED_Y):
            for x in range(IMAGES_DISPLAYED_X):
                self.setDisplayedImage(self.imageLabel[x][y], next(gen))

    def createNavBar(self):
        self.navBar_layout = QGridLayout()

        leftButton = QPushButton("<")
        leftButton.clicked.connect(self.leftClickCallback)
        self.navBar_layout.addWidget(leftButton, 0, 0)

        self.currentLineLabel = QLabel("0")
        self.navBar_layout.addWidget(self.currentLineLabel, 0, 1)

        rightButton = QPushButton(">")
        rightButton.clicked.connect(self.rightClickCallback)
        self.navBar_layout.addWidget(rightButton, 0, 2)

    def setDisplayedImage(self, label, pixelArray):
        image = createImage(pixelArray)
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
        label.repaint()
        self.currentLineLabel.setText(str(self.currentLine))

    def __init__(self):
        super().__init__()
        self.title = "Image Viewer"
        self.setWindowTitle(self.title)
        self.currentLine = 0
        overallLayout = QGridLayout()
        self.setLayout(overallLayout)
        self.createNavBar()

        overallLayout.addLayout(self.navBar_layout, 0, 0)
        self.imageLabel = [[0 for x in range(IMAGES_DISPLAYED_Y)] for y in range(IMAGES_DISPLAYED_X)]

        imageLayout = QGridLayout()
        gen = readCSV_Lines(0, IMAGES_DISPLAYED)
        for y in range(IMAGES_DISPLAYED_Y):
            for x in range(IMAGES_DISPLAYED_X):
                self.imageLabel[x][y] = QLabel(self)
                self.setDisplayedImage(self.imageLabel[x][y], next(gen))
                imageLayout.addWidget(self.imageLabel[x][y], x, y)

        overallLayout.addLayout(imageLayout, 1, 0)


def get_file_picker_dialog() -> QFileDialog:
    dialog = QFileDialog()
    dialog.setWindowTitle("Choose images to display")
    dialog.setFileMode(QFileDialog.AnyFile)
    dialog.setNameFilter("CSV files (*.csv)")
    return dialog


def main():
    app = QApplication(sys.argv)
    file_dialog = get_file_picker_dialog()
    if file_dialog.exec_():
        chosen_file = file_dialog.selectedFiles()[0]


if __name__ == "__main__":
    main()
