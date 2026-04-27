import os
import cv2
import numpy as np
from PIL import Image
from facewarehouse.exp_estimate import ExpEstimator
from utils import *
from config import conf
from tqdm import tqdm


class ExpErrorsFWH:
    def __init__(self):
        self.exp_estimator = ExpEstimator()

    def get_exp(self, img, ldms):
        ldms = np.array(ldms, dtype=np.float32)
        if ldms.shape != (5, 2):
            raise ValueError(f"Landmarks shape should be (5, 2), got {ldms.shape}")

        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        exp_rst = self.exp_estimator.get_exp(image, ldms)
        return exp_rst

    def calculate_exp_errors(self, swapped_face_root):
        files = sorted(os.listdir(swapped_face_root))
        pb = tqdm(files)
        dist_sum = 0
        cnt = 0

        for name in pb:
            swp_face = cv2.imread(swapped_face_root + name)
            src_name, tgt_name = parse_img_name(name)

            tgt_face = cv2.imread(conf.ori_face_root + tgt_name + '.png')
            h, w = tgt_face.shape[:2]
            swp_face_resize = cv2.resize(swp_face, (w, h))
            tgt_ldms = read_ldms(conf.ori_face_ldms_root + tgt_name + '.txt')

            swapped_exp = self.get_exp(swp_face_resize, tgt_ldms)
            target_exp = self.get_exp(tgt_face, tgt_ldms)

            angle_dist = l2_metric(swapped_exp, target_exp)
            dist_sum += angle_dist
            cnt += 1
            dist_avg = dist_sum / float(cnt)
            pb.set_description("{}/{}: {:.4f}".format(dist_sum, cnt, dist_avg))

        rst = dist_sum / float(cnt)
        return rst
