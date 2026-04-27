import torch
import numpy as np
from facewarehouse.network.resnet50_task import resnet50_use
from facewarehouse.preprocess_img import Preprocess
from facewarehouse.load_data import BFM


class ExpEstimator:
    def __init__(self):
        self.model = resnet50_use()
        self.model.load_state_dict(torch.load("./facewarehouse/network/th_model_params.pth"))
        self.model.eval()
        self.model.cuda()
        for param in self.model.parameters():
            param.requires_grad = False

        self.facemodel = BFM("./facewarehouse/BFM/mSEmTFK68etc.chj")
        self.facemodel.to_torch(is_torch=True, is_cuda=True)
        self.lm3D = self.facemodel.load_lm3d("./facewarehouse/BFM/similarity_Lm3D_all.mat")

    def get_exp(self, img, ldms):
        input_img_org, lm_new, transform_params = Preprocess(img, ldms, self.lm3D)
        input_img = input_img_org.astype(np.float32)
        input_img = torch.from_numpy(input_img).permute(0, 3, 1, 2).cuda()
        arr_coef = self.model(input_img)
        coef = torch.cat(arr_coef, 1).data.cpu().numpy()
        # ex_coeff = coef[:, 80:144]  # expression coeff of dim 64
        ex_coeff = coef[:, 80:109]  # expression 前29维度
        return ex_coeff[0]