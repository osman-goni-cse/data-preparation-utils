import sys
import time
import cv2

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, 
    QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QTextEdit
)

from cn_detector import CNDetector
from char_detector import CharDetector
from text_recognizer import TextRecognizer
from text_corrector import correct_container_number

# TextRecognizer Algo
REC_ALGO_1 = "ABINet" # main rec algorithm
REC_ALGO_2 = "CPPD" # auxiliary rec algorithm
USE_GPU = False

class InitAIModelThread(QThread):
    update_log_signal = pyqtSignal(str, str)
    models_loaded_signal = pyqtSignal(object, object, object, object)

    def run(self):
        try:
            self.update_log_signal.emit("Initializing AI Models ...", "info")
            cn_detector = CNDetector()
            self.update_log_signal.emit("CNDetector is ready.", "success")
            char_detector = CharDetector()
            self.update_log_signal.emit("CharDetector is ready.", "success")
            text_recognizer = TextRecognizer(algo=REC_ALGO_1, use_gpu=USE_GPU) 
            text_recognizer_2 = TextRecognizer(algo=REC_ALGO_2, use_gpu=USE_GPU)
            self.update_log_signal.emit("TextRecognizers are ready.", "success")
            self.models_loaded_signal.emit(cn_detector, char_detector, text_recognizer, text_recognizer_2)
        except Exception as e:
            error_message = f"Model loading failed: {str(e)}"
            self.update_log_signal.emit(error_message, "error")
            self.models_loaded_signal.emit(None, None, None, None)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Container Number Recognition")
        self.initUI()
        self.initAIModel()

    def initAIModel(self):
        self.init_ai_model_thread = InitAIModelThread()
        self.init_ai_model_thread.update_log_signal.connect(self.updateLog)
        self.init_ai_model_thread.models_loaded_signal.connect(self.modelsLoaded)
        self.init_ai_model_thread.start()

    def modelsLoaded(self, cn_detector, char_detector, text_recognizer, text_recognizer_2):
        if cn_detector is None or char_detector is None or text_recognizer is None or text_recognizer_2 is None:
            self.updateLog("Failed to load one or more models, please check.", "error")
        else:
            self.cn_detector = cn_detector
            self.char_detector = char_detector
            self.text_recognizer = text_recognizer
            self.text_recognizer_2 = text_recognizer_2
            self.updateLog("All AI models are loaded.", "info")
            self.open_button.setDisabled(False)

    def initUI(self):
        # Main layout container
        main_layout = QHBoxLayout()
        
        # Image label setup
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480) # Fixed size for the image display area
        self.image_label.setScaledContents(True) # Image will scale within the QLabel

        # Create a white background as a placeholder for the QLabel
        self.img_background = QPixmap(640, 480)
        self.img_background.fill(Qt.GlobalColor.lightGray)
        self.image_label.setPixmap(self.img_background)

        # Add the image label to the main layout
        main_layout.addWidget(self.image_label)

        # Side panel setup
        side_panel_layout = QVBoxLayout()
        side_panel_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align items to the top of the layout

        # Set a minimum width for the side panel
        side_panel_width = 300 # Adjust the width as needed
        side_panel = QWidget()
        side_panel.setMinimumWidth(side_panel_width)
        side_panel.setLayout(side_panel_layout)

        # Buttons setup
        self.open_button = QPushButton('Open Image')
        self.open_button.setDisabled(True)
        self.reset_button = QPushButton('Reset')
        side_panel_layout.addWidget(self.open_button)
        side_panel_layout.addWidget(self.reset_button)

        # Add a log printout text box to the side panel
        self.log_box = QTextEdit(self)
        self.log_box.setReadOnly(True) 
        self.log_box.setContentsMargins(5, 5, 5, 5)
        self.log_box.setStyleSheet("background-color: white;")
        self.log_box.setFixedSize(side_panel_width-20, 420)
        side_panel_layout.addWidget(self.log_box)

        # Add a stretch to ensure that everything is aligned to the top
        side_panel_layout.addStretch()

        # Add the side panel to the main layout
        main_layout.addWidget(side_panel)

        # Container widget setup
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Connect buttons to their respective slots
        self.open_button.clicked.connect(self.openImageFile)
        self.reset_button.clicked.connect(self.resetApplication)
    
    def updateLog(self, text, type):
        if type == "info":
            color = "blue"
        elif type == "success":
            color = "green"
        elif type == "error":
            color = "red"
        elif type == "warning":
            color = "orange"
        else:  # default to black
            color = "black"
        
        colored_text = f"<span style='color: {color};'>{text}</span>"
        self.log_box.append(colored_text)

    def openImageFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpeg *.jpg)")
        if fileName:
            self.image_path = fileName # Update the image path
            self.displayImage() # detect and display the result image

    def resetApplication(self):
        # Reset the image label
        self.image_label.setPixmap(self.img_background)
        self.log_box.clear()

    def getCroppedCN(self, image, boxes):
        # boxes: [[x1, y1, x2, y2, conf1, class],[...box2....],[...box3...],..., [...boxN...]]]
        # conf1 > conf2 > conf3 > ... > confN
        # class: 0 = CN, 1 = CN_ABC, 2 = CN_NUM, 3 = TS
        image_copy = image.copy()
        cropped_cn_image = None

        is_cn_detected = False
        cn_box = None
        is_cn_abc_detected = False
        cn_abc_box = None
        is_cn_num_detected = False
        cn_num_box = None
        is_ts_detected = False
        ts_box = None

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            if cls == 0 and is_cn_detected == False:
                is_cn_detected = True
                cn_box = box
            elif cls == 1 and is_cn_detected == False and is_cn_abc_detected == False:
                is_cn_abc_detected = True
                cn_abc_box = box
            elif cls == 2 and is_cn_detected == False and is_cn_num_detected == False:
                is_cn_num_detected = True
                cn_num_box = box
            elif cls == 3 and is_ts_detected == False:
                is_ts_detected = True
                ts_box = box
        ###################draw the bounding box on the image###################
        if is_cn_detected == True:
            x1, y1, x2, y2, conf, cls = cn_box
            cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        elif is_cn_abc_detected == True and is_cn_num_detected == True:
            x1, y1, x2, y2, conf, cls = cn_abc_box
            cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            x1, y1, x2, y2, conf, cls = cn_num_box
            cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if is_ts_detected == True:
            print("TS detected, ignore temporarily.")
            x1, y1, x2, y2, conf, cls = ts_box
            cv2.rectangle(image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 215, 255), 2) # orange
        ####################################################################### 
        if is_cn_detected == True:
            x1, y1, x2, y2, conf, cls = cn_box
            # crop the image based on the box coordinates with some extra padding
            # need to handle the case where the box is at the edge of the image
            p = 5
            if int(y1)-p < 0 or int(y2)+p > image.shape[0] or int(x1)-p < 0 or int(x2)+p > image.shape[1]:
                # at least one of the box coordinates is at the edge of the image
                # crop the image without padding
                cropped_cn_image = image[int(y1):int(y2), int(x1):int(x2)]
            else:
                # crop the image with padding
                cropped_cn_image = image[int(y1)-p:int(y2)+p, int(x1)-p:int(x2)+p]
            # first CN should have the highest confidence, so we return it. horizontal or vertical does not matter
            self.updateLog("One good CN line detected.", "default")
            print("CN detected, no stitching required, return directly.")
            return cropped_cn_image, image_copy
        elif is_cn_abc_detected == True and is_cn_num_detected == True:
            # CN_ABC
            x1, y1, x2, y2, conf, cls = cn_abc_box
            p = 5
            if int(y1)-p < 0 or int(y2)+p > image.shape[0] or int(x1)-p < 0 or int(x2)+p > image.shape[1]:
                cn_abc_box = [int(x1), int(y1), int(x2), int(y2), conf, cls]
            else:
                cn_abc_box = [int(x1)-p, int(y1)-p, int(x2)+p, int(y2)+p, conf, cls]
            # CN_NUM
            x1, y1, x2, y2, conf, cls = cn_num_box
            if int(y1)-p < 0 or int(y2)+p > image.shape[0] or int(x1)-p < 0 or int(x2)+p > image.shape[1]:
                cn_num_box = [int(x1), int(y1), int(x2), int(y2), conf, cls]
            else:
                cn_num_box = [int(x1)-p, int(y1)-p, int(x2)+p, int(y2)+p, conf, cls]
            
            print("Ready to stich CN_ABC & CN_NUM together.")
            # put the two boxes together
            # first, we need to check if the two boxes are horizontal or vertical
            is_box_cn_abc_horizontal = self.isBoxHorizontal(cn_abc_box)
            is_box_cn_num_horizontal = self.isBoxHorizontal(cn_num_box)

            x1, y1, x2, y2 = cn_abc_box[:4]
            img_abc = image[int(y1):int(y2), int(x1):int(x2)]
            x1, y1, x2, y2 = cn_num_box[:4]
            img_num = image[int(y1):int(y2), int(x1):int(x2)]

            if is_box_cn_abc_horizontal == True and is_box_cn_num_horizontal == True:
                print("Stitching two boxes horizontally.")
                # put cn_abc on left, cn_num on right
                h = max(img_abc.shape[0], img_num.shape[0])
                img_abc = self.add_padding(img_abc, h, img_abc.shape[1], 'vertical')
                img_num = self.add_padding(img_num, h, img_num.shape[1], 'vertical')
                stitched_cn_image = cv2.hconcat([img_abc, img_num])
                self.updateLog("Two horizontal CN lines detected.", "default")
                cv2.imwrite("temp_images/stitched_cn_image.jpg", stitched_cn_image)
                return stitched_cn_image, image_copy
            elif is_box_cn_abc_horizontal == False and is_box_cn_num_horizontal == False:
                print("Stitching two boxes vertically.")
                # put cn_abc on top, cn_num on bottom
                w = max(img_abc.shape[1], img_num.shape[1])
                img_abc = self.add_padding(img_abc, img_abc.shape[0], w, 'horizontal')
                img_num = self.add_padding(img_num, img_num.shape[0], w, 'horizontal')
                stitched_cn_image = cv2.vconcat([img_abc, img_num])
                self.updateLog("Two vertical CN lines detected.", "default")
                cv2.imwrite("temp_images/stitched_cn_image.jpg", stitched_cn_image)
                return stitched_cn_image, image_copy
            else:
                print("CN_ABC and CN_NUM are not in same direction, return None.")
                return None, image_copy
        elif is_cn_detected == False and is_cn_abc_detected == False and is_cn_num_detected == False:
            print("CN, CN_ABC and CN_NUM not detected, return None.")
            return None, image_copy
        else:
            print("CN not detected; CN_ABC or CN_NUM not detected, return None.")
            return None, image_copy

    def add_padding(self, img, target_height, target_width, direction):
        """
        Add black padding to an image to reach the target size.
        """
        if direction == 'horizontal':
            padding = (target_width - img.shape[1], 0)
        else:  # 'vertical'
            padding = (0, target_height - img.shape[0])

        padded_img = cv2.copyMakeBorder(img, 0, padding[1], 0, padding[0], cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_img

    def isBoxHorizontal(self, box):
        x1, y1, x2, y2, conf, cls = box
        if abs(x2-x1) > abs(y2-y1):
            return True
        else:
            return False

    def startWork(self, image):
        self.updateLog("Start det and rec...", "info")
        res_cndet = self.cn_detector.detect(image)
        # boxes: [[x1, y1, x2, y2, conf, class],[...box2....],[...box3...],..., [...boxN...]]]
        # class: 0 = CN, 1 = CN_ABC, 2 = CN_NUM, 3 = TS
        if len(res_cndet) == 0:
            self.updateLog("No CN/CN_ABC/CN_NUM/TS detected.", "default")
        else:
            # cropped_cn_image: cropped & stiched cn image, image: original image with bounding box
            cropped_cn_image, image = self.getCroppedCN(image, res_cndet)

            if cropped_cn_image is None:
                self.updateLog("No good container number detected.", "default")
                return image
            # detect the characters in the cropped image
            image_after_chardet, is_vertical, is_reassembled = self.char_detector.detect(cropped_cn_image)

            # recognize the characters in the cropped image
            res_rec = self.text_recognizer.rec(image_after_chardet)
            res_rec_2 = self.text_recognizer_2.rec(image_after_chardet)
            if len(res_rec) == 0:
                self.updateLog("No good container number recognized.", "default")
            else:
                #self.updateLog(f"{len(res_rec)} good CN recognized.", "default")
                # temporarily only use the first (1st conf) recognized container number
                cn_text, cn_conf = res_rec[0]
                if len(res_rec_2) != 0:
                    cn_text_2, cn_conf_2 = res_rec_2[0]
                self.updateLog(f"CN_1: {cn_text} ({cn_conf:.3f})", "default")
                self.updateLog(f"CN_2: {cn_text_2} ({cn_conf_2:.3f})", "default")
                # correct the recognized text
                corrected_cn_text = correct_container_number(cn_text, cn_text_2)
                
                if corrected_cn_text == "XXXX0000000":
                    self.updateLog("Wrong CN recognition.", "warning")
                    print("Wrong CN recognition.")
                    ################################
                    # if horizontal and reassembled cropped text image, try to recognize original cropped image again
                    if is_reassembled and is_vertical == False: 
                        self.updateLog("Try to recognize again...", "info")
                        print("Try to recognize again (horizontal and reassemble) ...")
                        res_rec = self.text_recognizer.rec(cropped_cn_image)
                        cn_text, cn_conf = res_rec[0]
                        res_rec_2 = self.text_recognizer_2.rec(cropped_cn_image)
                        cn_text_2, cn_conf_2 = res_rec_2[0]
                        self.updateLog(f"CN_1: {cn_text} ({cn_conf:.3f})", "default")
                        self.updateLog(f"CN_2: {cn_text_2} ({cn_conf_2:.3f})", "default")
                        corrected_cn_text = correct_container_number(cn_text, cn_text_2)
                        if corrected_cn_text == "XXXX0000000":
                            self.updateLog("Wrong CN recognition again.", "warning")
                            print("Wrong CN recognition again.")
                        else:
                            self.updateLog(f"Final CN: {corrected_cn_text}", "success")
                            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
                            cv2.putText(image, corrected_cn_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            print(f"Final CN: {corrected_cn_text}")
                    ################################
                else:
                    self.updateLog(f"Final CN: {corrected_cn_text}", "success")
                    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
                    cv2.putText(image, corrected_cn_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(f"Final CN: {corrected_cn_text}")

        return image
    
    def cv2_to_qImage(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        if pixmap.isNull():
            raise ValueError("Failed to convert QPixmap.")
        return pixmap

    def displayImage(self):
        image = cv2.imread(self.image_path)
        if image is None:
            self.updateLog(f"Image loaded error: {self.image_path}", "error")
        else:
            self.updateLog("------------------------------------------------", "default")
            self.updateLog("Image loaded successfully.", "success")
            print(f"Image loaded: {self.image_path}")
        
        # Display the original image
        pixmap = self.cv2_to_qImage(image)
        self.image_label.setPixmap(pixmap)

        st = time.time()
        image = self.startWork(image)
        et = time.time()
        self.updateLog(f"Total process time: {et-st:.3f}s", "info")
        print(f"Total process time: {et-st:.3f}s")
        # Display the result image
        pixmap = self.cv2_to_qImage(image)
        self.image_label.setPixmap(pixmap)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())