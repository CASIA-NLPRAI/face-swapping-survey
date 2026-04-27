import os
import cv2
from face_recognition.recognition import FaceRecognition
from tqdm import tqdm
from config import conf
from utils import *


class IDRetrieval:
    def __init__(self):
        self.fr = FaceRecognition()

    def calculate_id_retrieval(self, swapped_face_root):
        files = sorted(os.listdir(swapped_face_root))
        pb = tqdm(files)
        cnt = 0
        cnt_true = 0

        for name in pb:
            swp_face = cv2.imread(swapped_face_root + name)
            src_name, tgt_name = parse_img_name(name)
            vid = src_name.split('-')[0]

            tgt_face = cv2.imread(conf.ori_face_root + tgt_name + '.png')
            h, w = tgt_face.shape[:2]
            swp_face_resize = cv2.resize(swp_face, (w, h))
            tgt_ldms = read_ldms(conf.ori_face_ldms_root + tgt_name + '.txt')

            predict_vid = self.fr.check_id(swp_face_resize, tgt_ldms)

            if predict_vid == vid:
                cnt_true += 1

            cnt += 1
            ratio = cnt_true / float(cnt)
            pb.set_description("{}/{}: {:.4f}".format(cnt_true, cnt, ratio))

        rst = cnt_true / cnt
        return rst
