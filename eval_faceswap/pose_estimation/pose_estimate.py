import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
from pose_estimation.hopenet import Hopenet
from config import conf


class PoseEstimator(object):
    def __init__(self, pose_thr=30.):
        self.model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        self.model.load_state_dict(torch.load('./pose_estimation/hopenet_robust_alpha1.pkl'))
        self.model.to(conf.device)
        self.model.eval()

        self.transform = transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).cuda()
        self.pose_thr = pose_thr

    def get_angles(self, image, bboxs):
        face_num = len(bboxs)
        face_imgs = torch.zeros((face_num, 3, 224, 224))
        for j, boxes in enumerate(bboxs):
            if boxes is not None:
                # Crop face
                x_min = int(boxes[0])
                x_max = int(boxes[2])
                y_min = int(boxes[1])
                y_max = int(boxes[3])
                # Clip
                x_min = x_min if x_min > 0 else 0
                x_max = x_max if x_max < image.shape[1] else image.shape[1]
                y_min = y_min if y_min > 0 else 0
                y_max = y_max if y_max < image.shape[0] else image.shape[0]

                if not x_min < x_max or not y_min < y_max:
                    continue

                face_img = image[y_min:y_max, x_min:x_max]
                face_img = Image.fromarray(face_img)

                # Transform
                face_img = self.transform(face_img)
                img_shape = face_img.size()
                face_img = face_img.view(1, img_shape[0], img_shape[1], img_shape[2])
                face_imgs[j] = face_img

        face_imgs = face_imgs.cuda()
        yaws, pitchs, rolls = self.model(face_imgs)

        yaws_pred = F.softmax(yaws, dim=1)
        pitchs_pred = F.softmax(pitchs, dim=1)
        rolls_pred = F.softmax(rolls, dim=1)
        yaws_pred = torch.sum(yaws_pred.data * self.idx_tensor, dim=1) * 3 - 99
        pitchs_pred = torch.sum(pitchs_pred.data * self.idx_tensor, dim=1) * 3 - 99
        rolls_pred = torch.sum(rolls_pred.data * self.idx_tensor, dim=1) * 3 - 99
        yaws_pred = yaws_pred.detach().data.cpu().numpy().reshape(face_num, 1)
        pitchs_pred = pitchs_pred.detach().data.cpu().numpy().reshape(face_num, 1)
        rolls_pred = rolls_pred.detach().data.cpu().numpy().reshape(face_num, 1)
        angles = np.concatenate((yaws_pred, pitchs_pred, rolls_pred), axis=1)

        return angles