import os
import cv2
import numpy as np
from pose_estimation.pose_estimate import PoseEstimator
from utils import *
from config import conf
from tqdm import tqdm


class PoseErrors:
    def __init__(self):
        self.pose_estimator = PoseEstimator()

    def get_img_angles(self, img):
        img = cv2.resize(img, (224, 224))
        angles = self.pose_estimator.get_angles(img, np.array([[0, 0, 224, 224]]))
        return angles[0]

    def calculate_pose_errors(self, swapped_face_root):
        files = sorted(os.listdir(swapped_face_root))
        pb = tqdm(files)
        dist_sum = 0
        cnt = 0

        for name in pb:
            swp_face = cv2.imread(swapped_face_root + name)
            src_name, tgt_name = parse_img_name(name)
            tgt_face = cv2.imread(conf.ori_face_root + tgt_name + '.png')

            swapped_angles = self.get_img_angles(swp_face)
            target_angles = self.get_img_angles(tgt_face)

            angle_dist = l2_metric(swapped_angles, target_angles)
            dist_sum += angle_dist
            cnt += 1
            dist_avg = dist_sum / float(cnt)
            pb.set_description("{}/{}: {:.4f}".format(dist_sum, cnt, dist_avg))

        rst = dist_sum / float(cnt)
        return rst
