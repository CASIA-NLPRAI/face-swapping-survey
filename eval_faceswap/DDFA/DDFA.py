import torch
import torchvision.transforms as transforms
from DDFA.models import mobilenet_v1
import torch.backends.cudnn as cudnn
import cv2
import numpy as np


class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


class DDFA:

    def __init__(self):
        # 1. load pre-tained model
        self.model = self.__get_model()

        self.transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

        self.STD_SIZE = 120

    def __get_model(self):
        # 1. load pre-tained model
        checkpoint_fp = 'DDFA/models/phase1_wpdc_vdc.pth.tar'
        arch = 'mobilenet_1'
        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)
        if torch.cuda.is_available():
            cudnn.benchmark = True
            model = model.cuda()
        model.eval()
        return model

    def get_exp_param(self, img):
        # h, w = img.shape[:2]
        # img_bbox = [0, 0, h, w]
        # forward: one step
        img = cv2.resize(img, dsize=(self.STD_SIZE, self.STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            if torch.cuda.is_available():
                input = input.cuda()
            param = self.model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            # p, offset, alpha_shp, alpha_exp = _parse_param(param)
            alpha_exp = param[52:]

            return alpha_exp
