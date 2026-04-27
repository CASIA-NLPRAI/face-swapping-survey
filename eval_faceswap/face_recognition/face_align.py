import numpy as np
import cv2


class FaceAlign:
    def __init__(self):
        self.imgSize = [112, 96]
        self.coord5point = [[30.2946, 51.6963],  # 112x96的目标点
                            [65.5318, 51.6963],
                            [48.0252, 71.7366],
                            [33.5493, 92.3655],
                            [62.7299, 92.3655]]

    def __align_transformation(self, std_landmark, landmark):
        std_landmark = np.matrix(std_landmark).astype(np.float64)
        landmark = np.matrix(landmark).astype(np.float64)

        c1 = np.mean(std_landmark, axis=0)
        c2 = np.mean(landmark, axis=0)
        std_landmark -= c1
        landmark -= c2

        s1 = np.std(std_landmark)
        s2 = np.std(landmark)
        std_landmark /= s1
        landmark /= s2

        U, S, Vt = np.linalg.svd(std_landmark.T * landmark)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    def align(self, img, landmarks):
        trans_matrix = self.__align_transformation(landmarks, self.coord5point)  # Shape: 3 * 3
        trans_matrix = trans_matrix[:2]
        rot_img = cv2.warpAffine(img, trans_matrix, (img.shape[1], img.shape[0]))
        face = rot_img[0:self.imgSize[0], 0:self.imgSize[1]]
        face = cv2.resize(face, (self.imgSize[1], self.imgSize[0]))
        return face


face_align = FaceAlign()