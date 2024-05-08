
import argparse
import math
import time
from typing import List

import cv2
import numpy as np

from .utils import create_operators, build_post_process, transform
from utils import OrtInferSession, read_yaml



class Structure:
    def __init__(self, config):

        session = OrtInferSession(config)
        self.predictor, self.input_tensor, self.output_tensors, self.config = session, session.session.get_inputs()[0], None, None
        

        pre_process_list = [{
            'Resize': {
                'size': [800, 608]
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image']
            }
        }]
        postprocess_params = {
            'name': 'PicoDetPostProcess',
            "layout_dict_path": config['layout_dict_path'],
            "score_threshold": config['layout_score_threshold'],
            "nms_threshold": config['layout_nms_threshold'],
        }

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]

        if img is None:
            return None, 0

        img = np.expand_dims(img, axis=0)
        img = img.copy()

        preds, elapse = 0, 1
        starttime = time.time()

        # self.input_tensor.copy_from_cpu(img)
        pred = self.predictor(img)

        output_names = self.predictor.get_output_names()
        num_outs = int(len(output_names) / 2)

        preds = dict(boxes=pred[:num_outs ], boxes_num=pred[num_outs :])

        post_preds = self.postprocess_op(ori_im, img, preds)
        elapse = time.time() - starttime
        return post_preds, elapse

if __name__ == "__main__":
    config = read_yaml(r'structure/config.yaml')
    layout_predictor = Structure(config)