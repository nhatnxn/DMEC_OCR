import os
import sys
import cv2

from .tools.infer import TextDetector, TextClassifier

model_urls = {
    'det': {
        'ch':
        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar',
        'en':
        'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_ppocr_mobile_v2.0_det_infer.tar'
    },
    'rec': {
        'ch': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/ppocr_keys_v1.txt'
        },
        'en': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/en_dict.txt'
        },
        'french': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/french_dict.txt'
        },
        'german': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/german_dict.txt'
        },
        'korean': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/korean_dict.txt'
        },
        'japan': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/japan_dict.txt'
        },
        'chinese_cht': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/chinese_cht_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/chinese_cht_dict.txt'
        },
        'ta': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/ta_dict.txt'
        },
        'te': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/te_dict.txt'
        },
        'ka': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ka_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/ka_dict.txt'
        },
        'latin': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/latin_ppocr_mobile_v2.0_rec_infer.tar',
            'dict_path': 'ppocr/utils/dict/latin_dict.txt'
        },
        'arabic': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/arabic_ppocr_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/arabic_dict.txt'
        },
        'cyrillic': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/cyrillic_ppocr_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/cyrillic_dict.txt'
        },
        'devanagari': {
            'url':
            'https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/devanagari_ppocr_mobile_v2.0_rec_infer.tar',
            'dict_path': './ppocr/utils/dict/devanagari_dict.txt'
        }
    },
    'cls':
    'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'
}

def parse_args(mMain=True, add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    if mMain:
        parser = argparse.ArgumentParser(add_help=add_help)
        # params for prediction engine
        parser.add_argument("--use_gpu", type=str2bool, default=True)
        parser.add_argument("--ir_optim", type=str2bool, default=True)
        parser.add_argument("--use_tensorrt", type=str2bool, default=False)
        parser.add_argument("--gpu_mem", type=int, default=8000)

        # params for text detector
        parser.add_argument("--image_dir", type=str)
        parser.add_argument("--det_algorithm", type=str, default='DB')
        parser.add_argument("--det_model_dir", type=str, default=None)
        parser.add_argument("--det_limit_side_len", type=float, default=960)
        parser.add_argument("--det_limit_type", type=str, default='max')

        # DB parmas
        parser.add_argument("--det_db_thresh", type=float, default=0.3)
        parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
        parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
        parser.add_argument("--use_dilation", type=bool, default=False)
        parser.add_argument("--det_db_score_mode", type=str, default="fast")

        # EAST parmas
        parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
        parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
        parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

        # params for text recognizer
        parser.add_argument("--rec_algorithm", type=str, default='CRNN')
        parser.add_argument("--rec_model_dir", type=str, default=None)
        parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
        parser.add_argument("--rec_char_type", type=str, default='ch')
        parser.add_argument("--rec_batch_num", type=int, default=6)
        parser.add_argument("--max_text_length", type=int, default=25)
        parser.add_argument("--rec_char_dict_path", type=str, default=None)
        parser.add_argument("--use_space_char", type=bool, default=True)
        parser.add_argument("--drop_score", type=float, default=0.5)

        # params for text classifier
        parser.add_argument("--cls_model_dir", type=str, default=None)
        parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
        parser.add_argument("--label_list", type=list, default=['0', '180'])
        parser.add_argument("--cls_batch_num", type=int, default=6)
        parser.add_argument("--cls_thresh", type=float, default=0.9)

        parser.add_argument("--enable_mkldnn", type=bool, default=False)
        parser.add_argument("--use_zero_copy_run", type=bool, default=False)
        parser.add_argument("--use_pdserving", type=str2bool, default=False)

        parser.add_argument("--lang", type=str, default='ch')
        parser.add_argument("--det", type=str2bool, default=True)
        parser.add_argument("--rec", type=str2bool, default=True)
        parser.add_argument("--use_angle_cls", type=str2bool, default=False)
        return parser.parse_args()
    else:
        return argparse.Namespace(
            use_gpu=True,
            ir_optim=True,
            use_tensorrt=False,
            gpu_mem=8000,
            image_dir='',
            det_algorithm='DB',
            det_model_dir=None,
            det_limit_side_len=960,
            det_limit_type='max',
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=2.5,
            use_dilation=False,
            det_db_score_mode="fast",
            det_east_score_thresh=0.8,
            det_east_cover_thresh=0.1,
            det_east_nms_thresh=0.2,
            rec_algorithm='CRNN',
            rec_model_dir=None,
            rec_image_shape="3, 32, 320",
            rec_char_type='ch',
            rec_batch_num=6,
            max_text_length=25,
            rec_char_dict_path=None,
            use_space_char=True,
            drop_score=0.5,
            cls_model_dir=None,
            cls_image_shape="3, 48, 192",
            label_list=['0', '180'],
            cls_batch_num=6,
            cls_thresh=0.9,
            enable_mkldnn=False,
            use_zero_copy_run=False,
            use_pdserving=False,
            lang='ch',
            det=True,
            rec=True,
            use_angle_cls=False)


class PaddleDetection(object):
    def __init__(self, **kwargs):

        postprocess_params = parse_args(mMain=False, add_help=False)
        print(postprocess_params)
        postprocess_params.__dict__.update(**kwargs)
        self.use_angle_cls = postprocess_params.use_angle_cls
        # if postprocess_params.rec_char_dict_path is None:
        use_inner_dict = True
        lang = 'latin'
        postprocess_params.rec_char_dict_path = model_urls['rec'][lang]['dict_path']
        print(model_urls['rec'][lang]['dict_path'])
        self.text_detector = TextDetector(postprocess_params)
        print('--'*20)
        self.text_classifier = TextClassifier(postprocess_params)

    def detect_box(self, image):
        dt_boxes,_ = self.text_detector(image)
        _, cls_res,_ = self.text_classifier([image])
        # cls_res = None

        return dt_boxes, cls_res 
