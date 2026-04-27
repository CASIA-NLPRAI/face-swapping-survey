import os
import argparse

from eval_id_retrieval import IDRetrieval
from eval_id_similarity import IDSimilarity
from eval_pose_err import PoseErrors
from eval_exp_3ddfa import ExpErrors3DDFA
from eval_exp_facewarehouse import ExpErrorsFWH
from eval_FID import calculate_FID
from eval_SSIM import CalSSIM
from config import conf


def parse_args():
    parser = argparse.ArgumentParser(description='FaceSwap Evaluation')
    parser.add_argument('--methods', nargs='+', required=True,
                        help='List of method names to evaluate')
    parser.add_argument('--types', nargs='+',
                        default=['all_simple', 'all_cross_ethnicity', 'all_cross_attribute'],
                        help='Swap types to evaluate')
    parser.add_argument('--gpu', type=str, default='0',
                        help='CUDA visible device id')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    idr = IDRetrieval()
    ids = IDSimilarity()
    pose_err = PoseErrors()
    exp_3ddfa = ExpErrors3DDFA()
    exp_fwh = ExpErrorsFWH()
    cal_ssim = CalSSIM()

    method_list = args.methods
    swp_type_list = args.types

    os.makedirs(conf.root, exist_ok=True)

    for method_name in method_list:
        for swp_type in swp_type_list:
            rsts_save_txt = conf.root + f'eval_results_{method_name}_{swp_type}.txt'

            with open(rsts_save_txt, 'w') as rstf:
                swapped_face_root = conf.swap_data_root + method_name + '/' + swp_type + '/'
                print('{} {} start, path: {}'.format(method_name, swp_type, swapped_face_root))
                rstf.write('{} {} start, path: {}\n'.format(method_name, swp_type, swapped_face_root))

                # 1. ID Retrieval
                rst_idretr = idr.calculate_id_retrieval(swapped_face_root)
                print('{} {} id_retrieval result: {}'.format(method_name, swp_type, rst_idretr))
                rstf.write('{} {} id_retrieval result: {}\n'.format(method_name, swp_type, rst_idretr))

                # 2. ID Similarity
                rst_txt_idsimil = conf.result_txt_id_simil + '{}--{}.txt'.format(method_name, swp_type)
                rst_idsimil = ids.calculate_id_similarity(swapped_face_root, rst_txt_idsimil)
                print('{} {} id_similarity result: {}'.format(method_name, swp_type, rst_idsimil))
                rstf.write('{} {} id_similarity result: {}\n'.format(method_name, swp_type, rst_idsimil))

                # 3. Pose Error
                rst_poseer = pose_err.calculate_pose_errors(swapped_face_root)
                print('{} {} pose_err result: {}'.format(method_name, swp_type, rst_poseer))
                rstf.write('{} {} pose_err result: {}\n'.format(method_name, swp_type, rst_poseer))

                # 4. Expression 3DDFA
                rst_exp_3ddfa = exp_3ddfa.calculate_exp_errors(swapped_face_root)
                print('{} {} exp_3ddfa result: {}'.format(method_name, swp_type, rst_exp_3ddfa))
                rstf.write('{} {} exp_3ddfa result: {}\n'.format(method_name, swp_type, rst_exp_3ddfa))

                # 5. Expression FaceWarehouse
                rst_exp_fwh = exp_fwh.calculate_exp_errors(swapped_face_root)
                print('{} {} exp_fwh result: {}'.format(method_name, swp_type, rst_exp_fwh))
                rstf.write('{} {} exp_fwh result: {}\n'.format(method_name, swp_type, rst_exp_fwh))

                # 6. FID
                rst_fid = calculate_FID(swapped_face_root)
                print('{} {} cal_FID result: {}'.format(method_name, swp_type, rst_fid))
                rstf.write('{} {} cal_FID result: {}\n'.format(method_name, swp_type, rst_fid))

                # 7. SSIM
                rst_ssim = cal_ssim.calculate_ssim(swapped_face_root)
                print('{} {} cal_SSIM result: {}'.format(method_name, swp_type, rst_ssim))
                rstf.write('{} {} cal_SSIM result: {}\n'.format(method_name, swp_type, rst_ssim))
