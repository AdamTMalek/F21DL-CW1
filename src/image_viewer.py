import csv
import sys
import math

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtWidgets import QApplication, QLabel, QGridLayout, QPushButton, QWidget
from PyQt5.QtWidgets import QFileDialog


class ImageViewer(QWidget):
    def __init__(self, file):
        super().__init__()
        self.file = file
        self.image_size = self._get_image_size()
        self.images_rows = 5
        self.images_columns = 5
        self.total_images = self.images_rows * self.images_columns
        self.title = "Image Viewer"
        self.setWindowTitle(self.title)
        self.current_line = 0
        self.overall_layout = QGridLayout()
        self.setLayout(self.overall_layout)
        self.current_line_label = QLabel("0")
        self.navbar_layout = QGridLayout()
        self._create_navbar()
        self.overall_layout.addLayout(self.navbar_layout, 0, 0)
        self.image_label = [[0 for _ in range(self.images_columns)] for _ in range(self.images_rows)]
        self.image_layout = QGridLayout()

        self._display_images()

    def _get_image_size(self) -> int:
        with open(self.file, 'r') as file:
            line = file.readline()
            total_pixels = len(line.split(','))
            return int(math.sqrt(total_pixels))

    def _display_images(self):
        gen = self._read_csv_lines(0, self.total_images)
        for y in range(self.images_columns):
            for x in range(self.images_rows):
                self.image_label[x][y] = QLabel(self)
                self._set_displayed_image(self.image_label[x][y], next(gen))
                self.image_layout.addWidget(self.image_label[x][y], x, y)

        self.overall_layout.addLayout(self.image_layout, 1, 0)

    def _read_csv_lines(self, start: int, end: int):
        with open(self.file, "r") as file:
            data_reader = csv.reader(file)
            next(data_reader)  # skip header row
            for i in range(start):
                next(data_reader)
            for i in range(end - start):
                yield next(data_reader)

    def _create_image(self, pixel_array):
        image = QImage(self.image_size, self.image_size, QImage.Format_RGB32)
        for i, pixel in enumerate(pixel_array):
            pixelVal = int(float(pixel))
            image.setPixel(i % self.image_size, i // self.image_size, QColor(pixelVal, pixelVal, pixelVal, 255).rgb())
        return image

    def _left_click_callback(self):
        if self.current_line > 0:
            self.current_line -= self.total_images
        self._repopulate_images()

    def _right_click_callback(self):
        if self.current_line < 10000:
            self.current_line += self.total_images
        self._repopulate_images()

    def _repopulate_images(self):
        gen = self._read_csv_lines(self.current_line, self.current_line + self.total_images)
        for y in range(self.images_columns):
            for x in range(self.images_rows):
                self._set_displayed_image(self.image_label[x][y], next(gen))

    def _create_navbar(self):
        leftButton = QPushButton("<")
        leftButton.clicked.connect(self._left_click_callback)
        self.navbar_layout.addWidget(leftButton, 0, 0)

        self.navbar_layout.addWidget(self.current_line_label, 0, 1)

        rightButton = QPushButton(">")
        rightButton.clicked.connect(self._right_click_callback)
        self.navbar_layout.addWidget(rightButton, 0, 2)

    def _set_displayed_image(self, label, pixel_array):
        image = self._create_image(pixel_array)
        pixmap = QPixmap.fromImage(image)
        label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
        label.repaint()
        self.current_line_label.setText(str(self.current_line))


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
        image_viewer = ImageViewer(chosen_file)
        image_viewer.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
