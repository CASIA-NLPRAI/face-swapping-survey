import os
import torch


class Config(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_size = 256

    # Data paths - modify these to match your environment,
    # or set the corresponding environment variables.
    root = os.environ.get(
        'EVAL_RESULT_ROOT',
        '/path/to/swap/results/'
    )
    ori_data_root = os.environ.get(
        'EVAL_ORI_DATA_ROOT',
        '/path/to/original/faces/'
    )

    ori_face_root = ori_data_root + 'faces/'
    ori_face_ldms_root = ori_data_root + 'faces_landmarks/'
    swap_data_root = root
    result_txt_id_simil = root + 'result_txt_id_similarity/'


conf = Config()
