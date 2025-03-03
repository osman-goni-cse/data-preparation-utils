import os
import sys
import math
import time
import traceback
import cv2
import numpy as np

# set the root directory of PaddleOCR
__dir__  = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(__dir__ , 'PaddleOCR'))
import tools.infer.utility as utility # type: ignore
from ppocr.postprocess import build_post_process # type: ignore
from ppocr.utils.logging import get_logger # type: ignore

confidence_threshold = 0.6

logger = get_logger()

class TextRecognizer(object):
    def __init__(self, args=utility.parse_args(), algo="ABINet", use_gpu=False):
        args.use_gpu = use_gpu
        self.rec_batch_num = 6 # batch size for recognition
        self.rec_algorithm = algo

        postprocess_params = {}
        if self.rec_algorithm == 'ABINet':
            args.rec_model_dir =  './models/pdlocr_abinet_rec/'
            self.rec_image_shape = [3, 32, 128]
            postprocess_params = {
                'name': 'ABINetLabelDecode',
                "character_dict_path": './config/en_dict.txt',
                "use_space_char": False,
            }
        elif self.rec_algorithm in ["CPPD", "CPPDPadding"]:
            args.rec_model_dir =  './models/pdlocr_cppd_rec/'
            self.rec_image_shape = [3, 32, 100]
            postprocess_params = {
                'name': 'CPPDLabelDecode',
                "character_dict_path": './config/en_dict.txt',
                "use_space_char": False,
                "rm_symbol": True
            }

        self.postprocess_op = build_post_process(postprocess_params)
        self.postprocess_params = postprocess_params

        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'rec', logger)
        
        #self.warmup() # paddleocr does not need warmup manually actually
        print("{} ADV_TextRecognizer loaded and warmed up successfully.".format(self.rec_algorithm))

    def warmup(self):
        _dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        self.rec(_dummy_image)
    
    def resize_norm_img_svtr(self, img, image_shape):
        # for CPPD
        imgC, imgH, imgW = image_shape
        resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image

    def resize_norm_img_cppd_padding(self, img, image_shape, padding=True, interpolation=cv2.INTER_LINEAR):
        imgC, imgH, imgW = image_shape
        h = img.shape[0]
        w = img.shape[1]
        if not padding:
            resized_image = cv2.resize(img, (imgW, imgH), interpolation=interpolation)
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im

    def resize_norm_img_abinet(self, img, image_shape):
        imgC, imgH, imgW = image_shape

        resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image / 255.

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        resized_image = (resized_image - mean[None, None, ...]) / std[None, None, ...]
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')

        return resized_image
    
    def rec(self, img):
        img_list = [img] # input one image at 1 time now, can add more
        img_num = len(img_list)
        # calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0])) # w / h
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num

        st = time.time()

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)
            
            for ino in range(beg_img_no, end_img_no):
                if self.rec_algorithm == "CPPD":
                    norm_img = self.resize_norm_img_svtr(img_list[indices[ino]], self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm in ["CPPDPadding"]:
                    norm_img = self.resize_norm_img_cppd_padding(img_list[indices[ino]], self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "ABINet":
                    norm_img = self.resize_norm_img_abinet(img_list[indices[ino]], self.rec_image_shape)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
            
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            self.input_tensor.copy_from_cpu(norm_img_batch)
            self.predictor.run()
            outputs = []

            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            
            if len(outputs) != 1:
                preds = outputs
            else:
                preds = outputs[0]

            rec_result = self.postprocess_op(preds)

            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        # print(rec_res) # [('BCDU2107444', 0.9700266122817993)]

        good_result = []
        # print(f"{len(rec_res)} lines text recognized.")
        for line in rec_res:
            # line: ('TTNU8655846', 0.9970707297325134)
            text, confidence = line 
            confidence = round(confidence, 3)
            #print(f'rec text: {text}, confidence: {confidence}')
            if confidence > confidence_threshold:
                print(f'{self.rec_algorithm} rec text: {text}, {len(text)} chars, good confidence: {confidence}. Keeping.')
                good_result.append([text, confidence])
            else:
                print(f'{self.rec_algorithm} rec text: {text}, {len(text)} chars, low confidence: {confidence}. Ignoring.')
    
        #print("{} rec time: {}s".format(self.rec_algorithm, total_time.__round__(3)))
        print(f"{len(good_result)} lines text recognized and returned. Took time: {time.time() - st:.3f}s")
        #print(good_result) # [['BCDU2107444', 0.9700266122817993]]
        return good_result
"""
def main():
    text_recognizer = TextRecognizer(algo="ABINet")
    img = cv2.imread('./test_images/other1.jpg')
    try:
        rec_res = text_recognizer.rec(img)
    except Exception as E:
        #logger.info(traceback.format_exc())
        #logger.info(E)
        print(traceback.format_exc())
        print(E)
        exit()
    
    print("Predicts: {}".format(rec_res[0]))

if __name__ == "__main__":
    main()
"""