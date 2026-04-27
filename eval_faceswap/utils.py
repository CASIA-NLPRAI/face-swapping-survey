import re
import numpy as np
import cv2


def parse_img_name(img_name):
    name = img_name[:-4]
    str_list = name.split('-')

    if len(str_list) == 4:
        src_name = '-'.join(str_list[:2])
        tgt_name = '-'.join(str_list[2:])
    elif len(str_list) == 3:
        middle = str_list[1]
        match = re.match(r'(.+)_(\d{4})$', middle)
        if match:
            src_suffix = match.group(1)
            tgt_prefix = match.group(2)
            src_name = f"{str_list[0]}-{src_suffix}"
            tgt_name = f"{tgt_prefix}-{str_list[2]}"
        else:
            raise ValueError(f"Cannot parse RAFSwap format: {img_name}")
    else:
        raise ValueError(f"Unexpected image name format (expected 3 or 4 '-' separators): {img_name}")

    return src_name, tgt_name


def read_ldms(path):
    ldms_list = []
    with open(path, 'r') as txt_file:
        for line in txt_file.readlines():
            line_str = line.strip('\n').split()
            if len(line_str) >= 2:
                ldms_list.append([float(line_str[0]), float(line_str[1])])
    return np.array(ldms_list, dtype=np.float32)


def read_img(path, is_gray=False):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def l2_metric(x1, x2):
    return np.linalg.norm(x1 - x2)
