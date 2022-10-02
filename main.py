import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
from enum import Enum
import random
import warnings
import cv2
import os
import numpy as np
from numpynet import *
import matplotlib.pyplot as plt
import matplotlib.image
import time # For letting user see visualizations, not fake loading screens because im not fake im REAL

'''
DO NOT RUN IN SHELL!!

If you run it in the terminal, it will show ALL errors. IDLE Shell does not show all errors with Qt.

Icons credit:
Bharat Icons
https://www.flaticon.com/authors/bharat-icons
'''
fashion_mnist_labels = {
      0: 'T-shirt/top',
      1: 'Trouser',
      2: 'Pullover',
      3: 'Dress',
      4: 'Coat',
      5: 'Sandal',
      6: 'Shirt',
      7: 'Sneaker',
      8: 'Bag',
      9: 'Ankle boot'
}
'''
https://stackoverflow.com/questions/19902183/qimage-to-numpy-array-using-pyside
# Customized one of the functions from one of the stackoverflow answers (the one that worked)
Did that but with alpha channel removed, and options for swapping red and blue channels and for
working with grayscale images. Need to get numpy array format to do OpenCV preprocessing and neural
net inference on image. Then, convert it back to a QImage after each step to visualize the preprocessing
to the user!
'''
def QImageToCvMat(image, rgbSwapped=False):
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    if rgbSwapped:
        image = image.rgbSwapped()
        
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
    return arr

def CvMatToQImage(arr, rgbSwapped=False, grayscale=False):
    if grayscale:
        if rgbSwapped:
            warnings.warn("You are confused. Set rgbSwapped AND grayscale to True on the same image?????")
        height, width = arr.shape
        bytesPerLine = width
        image = QtGui.QImage(arr.data, width, height, bytesPerLine, QtGui.QImage.Format_Grayscale8)
    else:
        height, width, channel = arr.shape
        bytesPerLine = width * 3
        image = QtGui.QImage(arr.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        if rgbSwapped:
            image = image.rgbSwapped()
    return image

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Drawing Classifier")
        
        self.canvas = Canvas(400, 400)
        self.color_picker_button = ColorPickerButton(self.canvas)
        self.clear_canvas_button = ClearCanvasButton(self.canvas)

        pen_type_buttons_layout = QtWidgets.QVBoxLayout()
        for ptype in PenType:
            icon = QtGui.QIcon('Assets/' + f'{ptype.name}' + '.png')
            button = PenTypeButton(self.canvas, ptype, icon)
            pen_type_buttons_layout.addWidget(button)

        self.prediction_viewer = PredictionViewer()
        self.evaluator = NNEvaluator(self.canvas, self.prediction_viewer)
        self.evaluate_drawing_button = EvaluateDrawingButton(self.canvas, self.evaluator)
        
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.addLayout(pen_type_buttons_layout)
        button_layout.addWidget(self.color_picker_button)
        button_layout.addWidget(self.clear_canvas_button)
        button_layout.addWidget(self.evaluate_drawing_button)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addWidget(self.canvas)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.prediction_viewer)

        framed_layout = QtWidgets.QVBoxLayout()
        framed_layout.addWidget(TitleText())
        framed_layout.addWidget(HeaderText())
        framed_layout.addLayout(main_layout)
        
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(framed_layout)
        self.setCentralWidget(central_widget)
        
        # KNOWN BUG: This breaks the canvas
        #self.setFixedSize(640, 480)

class PenType(Enum):

    ERASER = 0
    PENCIL = 1 # Pencil bc only Jamie uses pen
    SPRAY = 2
    FILL = 3

class Canvas(QtWidgets.QLabel):

    def __init__(self, width, height):
        super().__init__()
        pixmap = QtGui.QPixmap(width, height)
        self.setPixmap(pixmap)

        # Pen properties & vars
        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor("#000000")
        self.pen_size = 5
        self.pen_type = PenType.PENCIL
        self.spray_particles = 20 * self.pen_size
        self.background_color = QtGui.QColor("#ffffff")
        self.isBusy = False

        #Fill canvas with background color initially
        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        painter.fillRect(0, 0, self.pixmap().width(), self.pixmap().height(), self.background_color)

    def set_pen_color(self, color):
        if not isinstance(color, QtGui.QColor):
            warnings.warn("Attempt to set pen color without converting to QColor object.")
            color = QtGui.QColor(color)
        self.pen_color = color

    def set_pen_size(self, size):
        self.pen_size = size
        self.spray_particles = 20 * size

    def mouseMoveEvent(self, e):
        if self.isBusy:
            return
        
        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        
        if self.pen_type == PenType.PENCIL:
            if self.last_x is None:
                self.last_x = e.x()
                self.last_y = e.y()
                return
            
            p.setWidth(self.pen_size)
            p.setColor(self.pen_color)
            painter.setPen(p)
            
            painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
            
        elif self.pen_type == PenType.SPRAY:
            p.setWidth(1)
            p.setColor(self.pen_color)
            painter.setPen(p)
            for n in range(self.spray_particles):
                xo = random.gauss(0, self.pen_size)
                yo = random.gauss(0, self.pen_size)
                painter.drawPoint(int(e.x() + xo), int(e.y() + yo))
                
        elif self.pen_type == PenType.ERASER:
            if self.last_x is None:
                self.last_x = e.x()
                self.last_y = e.y()
                return
            
            p.setWidth(self.pen_size)
            p.setColor(QtGui.QColor("#ffffff"))
            
            painter.setPen(p)
            painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
            
        elif self.pen_type == PenType.FILL:
            self.isBusy = True
            # https://www.pythonguis.com/faq/implementing-qpainter-flood-fill-pyqt5pyside/
            x = e.x()
            y = e.y()
            image = self.pixmap().toImage()
            w, h = image.width(), image.height()
            target_color = image.pixel(x, y)
            have_seen = set()
            queue = [(x, y)]
            
            def get_cardinal_points(have_seen, center_pos):
                points = []
                cx, cy = center_pos
                for x, y in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    xx, yy = cx + x, cy + y
                    if (xx >= 0 and xx < w and yy >= 0 and yy < h and (xx, yy) not in have_seen):
                        points.append((xx, yy))
                        have_seen.add((xx, yy))

                return points

            p.setColor(self.pen_color)
            painter.setPen(p)
            while queue:
                x, y = queue.pop()
                if image.pixel(x, y) == target_color:
                    painter.drawPoint(QtCore.QPoint(x, y))
                    queue[0:0] = get_cardinal_points(have_seen, (x, y))
            
        else:
            warnings.warn("Pen Type not recongized.")

        self.isBusy = False
        painter.end()  
        self.update()
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x, self.last_y = None, None

class ColorPickerButton(QtWidgets.QPushButton):

    def __init__(self, canvas_obj):
        super().__init__()
        self.color_picker = QtWidgets.QColorDialog()
        self.canvas_obj = canvas_obj
        self.setText("Change pen color")
        self.clicked.connect(self.on_clicked)
    
    def on_clicked(self):
        color = self.color_picker.getColor()
        if color.isValid():
            self.canvas_obj.set_pen_color(color)

class ClearCanvasButton(QtWidgets.QPushButton):

    def __init__(self, canvas_obj):
        super().__init__()
        self.canvas_obj = canvas_obj
        self.setText("Clear")
        self.clicked.connect(self.on_clicked)

    def on_clicked(self):
        painter = QtGui.QPainter(self.canvas_obj.pixmap())
        p = painter.pen()
        painter.fillRect(0, 0, self.canvas_obj.pixmap().width(), self.canvas_obj.pixmap().height(), self.canvas_obj.background_color)
        self.canvas_obj.update()

class PenTypeButton(QtWidgets.QPushButton):

    def __init__(self, canvas_obj, pen_type, icon=None):
        super().__init__()
        self.setFixedSize(QtCore.QSize(50, 50))
        self.pen_type = pen_type
        self.canvas_obj = canvas_obj
        if icon:
            self.setIcon(icon)
            self.setIconSize(QtCore.QSize(40, 40))
            
        self.clicked.connect(self.on_clicked)

    def on_clicked(self):
        self.canvas_obj.pen_type = self.pen_type

class TitleText(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        self.setText("Neural Network Drawing Classifier")
        self.setFont(QtGui.QFont('Helvetica', 30))
        self.setAlignment(Qt.AlignCenter)

class HeaderText(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        path = "Assets/header.txt"
        text = ""
        if os.path.isfile(path):
            text_file = open(path, "r")
            text = text_file.read()
            text_file.close()
        self.setWordWrap(True)
        self.setText(text)
        self.setFont(QtGui.QFont('Helvetica', 14))
        self.setAlignment(Qt.AlignCenter)

# SUCH A PAIN IN THE BUTTOCKS TO DO MULTITHREADING DHFDSHFDS
class NNEvaluator():
    
    def __init__(self, canvas_obj, prediction_viewer):
        self.canvas_obj = canvas_obj
        self.prediction_viewer = prediction_viewer
        self.model = Model.load("test.model")
        self.worker = self.Worker(self.canvas_obj, self.prediction_viewer, self.model)

    def newWorker(self):
        self.worker = self.Worker(self.canvas_obj, self.prediction_viewer, self.model)
        
    class Worker(QtCore.QObject):
        
        finished = QtCore.pyqtSignal()
        def __init__(self, canvas_obj, prediction_viewer, model):
            super().__init__()
            self.canvas_obj = canvas_obj
            self.prediction_viewer = prediction_viewer
            self.model = model
        
        def run(self):
            self.canvas_obj.isBusy = True
            # Img proc
            pixmap_save = self.canvas_obj.pixmap().copy()
            q_image = self.canvas_obj.pixmap().toImage()
            
            
            image = QImageToCvMat(q_image)
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            resized_image = cv2.resize(grayscale_image, (28, 28))
            inverted_image = 255 - resized_image
            # https://stackoverflow.com/questions/19363293/whats-the-fastest-way-to-increase-color-image-contrast-with-opencv-in-python-c
            # https://stackoverflow.com/questions/48406578/adjusting-contrast-of-image-purely-with-numpy
            min_val = np.min(inverted_image)
            max_val = np.max(inverted_image)
            LUT = np.zeros(256,dtype=np.uint8)
            LUT[min_val:max_val+1]=np.linspace(start=0,stop=255,num=(max_val-min_val)+1,endpoint=True,dtype=np.uint8)
            brightened_image = LUT[inverted_image]
            image_data = (brightened_image.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
            confidences = self.model.predict(image_data)
            predictions = self.model.output_layer_activation.predictions(confidences)
            prediction = fashion_mnist_labels[predictions[0]]
            
            grayscale_qimg = CvMatToQImage(grayscale_image, grayscale=True)
            resized_qimg = CvMatToQImage(resized_image, grayscale=True)
            inverted_qimg = CvMatToQImage(inverted_image, grayscale=True)
            brightened_qimg = CvMatToQImage(brightened_image, grayscale=True)
            
            self.canvas_obj.setPixmap(QtGui.QPixmap(grayscale_qimg))
            time.sleep(2)
            
            
            self.canvas_obj.setPixmap(QtGui.QPixmap(resized_qimg).scaled(self.canvas_obj.width(),
                                                                         self.canvas_obj.height(), QtCore.Qt.KeepAspectRatio))
            time.sleep(2)
            self.canvas_obj.setPixmap(QtGui.QPixmap(inverted_qimg).scaled(self.canvas_obj.width(),
                                                                         self.canvas_obj.height(), QtCore.Qt.KeepAspectRatio))
            time.sleep(2)
            self.canvas_obj.setPixmap(QtGui.QPixmap(brightened_qimg).scaled(self.canvas_obj.width(),
                                                                         self.canvas_obj.height(), QtCore.Qt.KeepAspectRatio))
            time.sleep(2)
            self.prediction_viewer.prediction.setText(prediction)
            for i in range(10):
                label = self.prediction_viewer.grid.itemAtPosition(i, 1).widget()
                label.setText(f'{confidences[0][i]:.2%}')
            print(f"NNEvaluator: Computed forward pass. prediction={prediction}")
            self.canvas_obj.setPixmap(pixmap_save)
            self.canvas_obj.isBusy = False
            self.finished.emit()

class EvaluateDrawingButton(QtWidgets.QPushButton):

    def __init__(self, canvas_obj, evaluator):
        super().__init__()
        self.canvas_obj = canvas_obj
        self.evaluator = evaluator
        self.worker = evaluator.worker
        self.setText("Evaluate Drawing")     
        self.clicked.connect(self.on_clicked)
    
    def on_clicked(self):
        # https://realpython.com/python-pyqt-qthread/
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_worker_finished(self):
        self.evaluator.newWorker()
        self.worker = self.evaluator.worker

class PredictionViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.num_predictions = 10
        self.title = QtWidgets.QLabel()
        self.title.setText("Guess:")
        self.title.setFont(QtGui.QFont('Helvetica', 14))
        self.title.setAlignment(Qt.AlignCenter)
        
        self.prediction = QtWidgets.QLabel()
        self.prediction.setText("None")
        self.prediction.setFont(QtGui.QFont('Helvetica', 18))
        self.prediction.setAlignment(Qt.AlignCenter)
        
        layout = QtWidgets.QVBoxLayout()
        self.grid = QtWidgets.QGridLayout()

        layout.addWidget(self.title)
        layout.addWidget(self.prediction)
        layout.addLayout(self.grid)
        
        self.setLayout(layout)
        for r in range(self.num_predictions):
            label = QtWidgets.QLabel()
            label.setText(fashion_mnist_labels.get(r))
            value = QtWidgets.QLabel()
            value.setText("00.00%")
            self.grid.addWidget(label, r, 0)
            self.grid.addWidget(value, r, 1)

#class BrushSizeWidget(

# https://www.youtube.com/watch?v=g_wlZ9IhbTs
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    print(f"App window running. Size: {window.size()}")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
