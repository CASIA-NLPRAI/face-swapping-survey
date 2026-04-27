import torch
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
from PIL import Image
from face_recognition import net
from face_recognition.face_align import face_align
import numpy as np
import cv2
from config import conf


class FaceRecognition():
    def __init__(self):
        self.model = net.sphere(type=20, is_gray=False)
        self.model.load_state_dict(torch.load('./face_recognition/CosFace_ACC99.28.pth'))
        self.model.to(conf.device)
        self.model.eval()

        self.features_dict = np.load('./id_prepare/features_dict.npy', allow_pickle=True).item()
        self.features = np.array(list(self.features_dict.values()))
        self.str_vids = np.array(list(self.features_dict.keys()))

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])

    def check_id(self, img, ldms):
        """
        img: 输入为cv2读取的BGR格式

        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = face_align.align(img, ldms)
        face_ = Image.fromarray(face)
        face, face_ = self.transform(face), self.transform(F.hflip(face_))
        face, face_ = face.unsqueeze(0), face_.unsqueeze(0)
        face_data = torch.cat((face, face_), 0).to(conf.device)

        output = self.model(face_data)
        output = output.data.cpu().numpy()
        fe_1 = output[::2]
        fe_2 = output[1::2]
        fe = np.hstack((fe_1, fe_2)).reshape((1024,))
        ft = fe / np.linalg.norm(fe)

        cos_sims = np.dot(self.features, ft.T)

        # top1
        index = np.argmax(cos_sims)
        predict_vid = self.str_vids[index]

        return predict_vid

    def get_feature_norm(self, img, ldms):
        """
        img: 输入为cv2读取的BGR格式

        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = face_align.align(img, ldms)
        face_ = Image.fromarray(face)
        face, face_ = self.transform(face), self.transform(F.hflip(face_))
        face, face_ = face.unsqueeze(0), face_.unsqueeze(0)
        face_data = torch.cat((face, face_), 0).to(conf.device)

        output = self.model(face_data)
        output = output.data.cpu().numpy()
        fe_1 = output[::2]
        fe_2 = output[1::2]
        fe = np.hstack((fe_1, fe_2)).reshape((1024,))
        ft = fe / np.linalg.norm(fe)
        return ft