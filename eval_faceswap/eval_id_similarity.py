import os
import cv2
import numpy as np
from face_recognition.recognition import FaceRecognition
from tqdm import tqdm
from config import conf
from utils import *


class IDSimilarity:
    def __init__(self):
        self.fr = FaceRecognition()

    def calculate_id_similarity(self, swapped_face_root, id_sims_save_txt):
        files = sorted(os.listdir(swapped_face_root))
        pb = tqdm(files)
        cnt = 0
        sim_sum = 0

        os.makedirs(os.path.dirname(id_sims_save_txt), exist_ok=True)

        with open(id_sims_save_txt, 'w') as txt_file:
            for name in pb:
                swp_face = cv2.imread(swapped_face_root + name)
                src_name, tgt_name = parse_img_name(name)

                tgt_face = cv2.imread(conf.ori_face_root + tgt_name + '.png')
                h, w = tgt_face.shape[:2]
                swp_face_resize = cv2.resize(swp_face, (w, h))
                tgt_ldms = read_ldms(conf.ori_face_ldms_root + tgt_name + '.txt')

                src_ldms = read_ldms(conf.ori_face_ldms_root + src_name + '.txt')
                src_face = cv2.imread(conf.ori_face_root + src_name + '.png')

                swp_ft_norm = self.fr.get_feature_norm(swp_face_resize, tgt_ldms)
                src_ft_norm = self.fr.get_feature_norm(src_face, src_ldms)

                cos_sim = np.dot(src_ft_norm, swp_ft_norm)
                txt_file.write('{} {}\n'.format(name, cos_sim))

                sim_sum += cos_sim
                cnt += 1
                sim_avg = sim_sum / float(cnt)
                pb.set_description("{:.6f}/{}: {:.6f}".format(sim_sum, cnt, sim_avg))

        rst = sim_sum / float(cnt)
        return rst
