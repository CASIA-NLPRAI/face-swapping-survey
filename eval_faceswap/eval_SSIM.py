import torch
import torchvision.transforms as TF
from pytorch_msssim import ssim
from tqdm import tqdm
from PIL import Image
from config import conf
from utils import *


class CalSSIM:
    def __init__(self):
        self.transform = TF.Compose([
            TF.Resize((conf.img_size, conf.img_size)),
            TF.ToTensor(),
            TF.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def calculate_ssim(self, swapped_face_root):
        files = sorted(os.listdir(swapped_face_root))
        pb = tqdm(files)
        n = len(files)
        ssim_sum = 0.0
        cnt = 0

        for name in pb:
            swp_face = Image.open(swapped_face_root + name)
            src_name, tgt_name = parse_img_name(name)
            src_face = Image.open(conf.ori_face_root + src_name + '.png')

            swp_data = self.transform(swp_face).unsqueeze(0)
            src_data = self.transform(src_face).unsqueeze(0)

            swp_data = (swp_data + 1) / 2
            src_data = (src_data + 1) / 2

            swp_data = swp_data.to(conf.device)
            src_data = src_data.to(conf.device)

            ssim_val = ssim(swp_data, src_data, data_range=1, size_average=False)
            ssim_sum += ssim_val.cpu().numpy()[0]

            cnt += 1
            ratio = ssim_sum / cnt
            pb.set_description("{:.6f}/{}: {:.6f}".format(ratio, cnt, ratio))

        rst = ssim_sum / n
        return rst
