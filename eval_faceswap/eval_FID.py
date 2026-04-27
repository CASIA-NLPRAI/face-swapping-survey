from pytorch_fid_new import fid_score
from config import conf


def calculate_FID(swapped_face_root, img_size=conf.img_size, batch_size=256):
    dim = 2048
    paths = [conf.ori_face_root, swapped_face_root]
    fid_value = fid_score.calculate_fid_given_paths(paths,
                                                    img_size=img_size,
                                                    batch_size=batch_size,
                                                    device=conf.device,
                                                    dims=dim)
    return fid_value
