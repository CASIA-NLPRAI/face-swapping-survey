import os
import cv2
import numpy as np


def get_img_paths(img_1w_root, img_1w_landmarks_root):
    images = []
    for img_name in os.listdir(img_1w_root):
        name_str = img_name.split('_')
        pid = int(name_str[1])
        img_path = img_1w_root + img_name
        landmark_path = img_1w_landmarks_root + img_name[:-4] + '.txt'
        images.append([pid, img_path, landmark_path])
    return images


def read_ldms(path):
    ldms_list = []
    with open(path, 'r') as txt_file:
        for line in txt_file.readlines():
            line_str = line.strip('\n').split(' ')
            ldms_list.append([float(line_str[0]), float(line_str[1])])
    return np.array(ldms_list)


def read_img(path, is_gray):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 模型输入为RGB格式数据
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def normalize_img_gray(img):
    image = img.copy()
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 128
    return image


def load_img(img, is_gray):
    if is_gray:
        return normalize_img_gray(img)
    else:
        b, g, r = cv2.split(img)
        bl = normalize_img_gray(b)
        gl = normalize_img_gray(g)
        rl = normalize_img_gray(r)
        image = np.concatenate((bl, gl, rl), axis=1)
        return image


def cosin_metric(x1, x2):
    a = np.linalg.norm(x1)
    xx1 = x1 / a
    aa = np.linalg.norm(xx1)
    b = np.linalg.norm(x2)
    xx2 = x2/b
    bb = np.linalg.norm(xx2)
    c = np.dot(x1, x2)
    cc = np.dot(xx1, xx2)
    rst1 = c / (a * b)
    rst2 = cc / (aa * bb)
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cosin_similarity(f1, f2):
    return f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)


def cosin_dist(x1, x2):
    sim = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    return 1-sim


def l2_metric(x1, x2):
    return np.linalg.norm(x1 - x2)