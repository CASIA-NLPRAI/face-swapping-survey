import os.path

import torch
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from PIL import Image
from face_recognition import net
from face_recognition.face_align import face_align
from config import conf
import numpy as np
from tqdm import tqdm
from utils import *


class PrepareFeatures():
    def __init__(self):
        self.model = net.sphere(type=20, is_gray=False)
        self.model.load_state_dict(torch.load('./face_recognition/CosFace_ACC99.28.pth'))
        self.model.to(conf.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def save_features(self, batch_size=10):
        csv_list = np.genfromtxt('./anno.csv', delimiter=",", dtype=str, encoding='utf-8')
        anno_list = csv_list[1:]
        simple_all = anno_list[0::2]

        images = None
        labels = []
        feature_dict = {}

        for i, elem in enumerate(tqdm(simple_all)):
            video_name = elem[0]
            vid = video_name.split('-')[0]
            img_name = vid + '-0_0.png'
            img = read_img(conf.ori_face_root + img_name)
            ldms = read_ldms(conf.ori_face_ldms_root + vid + '-0_0.txt')

            face = face_align.align(img, ldms)
            face_ = Image.fromarray(face)
            face, face_ = self.transform(face), self.transform(F.hflip(face_))
            face, face_ = face.unsqueeze(0), face_.unsqueeze(0)
            face_data = torch.cat((face, face_), 0)

            if images is None:
                images = face_data
            else:
                images = torch.cat((images, face_data), 0)

            labels.append(vid)

            if (i + 1) % batch_size == 0 or i == len(simple_all) - 1:
                data = images.to(conf.device)
                output = self.model(data)
                output = output.data.cpu().numpy()

                fe_1 = output[::2]
                fe_2 = output[1::2]
                feature = np.hstack((fe_1, fe_2))

                for fe, l in zip(feature, labels):
                    ft = fe / np.linalg.norm(fe)
                    feature_dict[l] = ft

                images = None
                labels = []

        save_path = './id_prepare/'
        os.makedirs(save_path, exist_ok=True)
        np.save(save_path + 'features_dict.npy', feature_dict)


if __name__ == '__main__':
    pf = PrepareFeatures()
    pf.save_features()
