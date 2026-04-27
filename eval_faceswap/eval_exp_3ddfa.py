import os
import cv2
import numpy as np
from DDFA.DDFA import DDFA
from utils import *
from config import conf
from tqdm import tqdm


class ExpErrors3DDFA:
    def __init__(self):
        self.dffa = DDFA()

    def calculate_exp_errors(self, swapped_face_root):
        files = sorted(os.listdir(swapped_face_root))
        pb = tqdm(files)
        dist_sum = 0
        cnt = 0

        for name in pb:
            swp_face = cv2.imread(swapped_face_root + name)
            src_name, tgt_name = parse_img_name(name)
            tgt_face = cv2.imread(conf.ori_face_root + tgt_name + '.png')

            swapped_exp = self.dffa.get_exp_param(swp_face)
            target_exp = self.dffa.get_exp_param(tgt_face)

            angle_dist = l2_metric(swapped_exp, target_exp)
            dist_sum += angle_dist
            cnt += 1
            dist_avg = dist_sum / float(cnt)
            pb.set_description("{}/{}: {:.4f}".format(dist_sum, cnt, dist_avg))

        rst = dist_sum / float(cnt)
        return rst
