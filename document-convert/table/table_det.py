
import argparse
import copy
import math
import time
from typing import List

import cv2
import numpy as np


from .utils import TableMatch, create_operators, build_post_process, sorted_boxes, transform
from utils import OrtInferSession, read_yaml


def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
    #     print(shape)
    h, w, c = shape
    tmp_x0 = x0 - pix
    tmp_x1 = x1 + pix
    tmp_y0 = y0 - pix
    tmp_y1 = y1 + pix
    x0_ = tmp_x0 if tmp_x0 >= 0 else 0
    x1_ = tmp_x1 if tmp_x1 <= w else w
    y0_ = tmp_y0 if tmp_y0 >= 0 else 0
    y1_ = tmp_y1 if tmp_y1 <= h else h
    return x0_, y0_, x1_, y1_


class RebuildTable(object):
    def __init__(self, args, det = None, rec = None):
        self.args = args

        self.text_detector = det
        self.text_recognizer = rec
        self.table_structurer = table(args)
        
        self.match = TableMatch(filter_ocr_result=True)

    def __call__(self, img, return_ocr_result_in_table=False):
        result = dict()
        time_dict = {'det': 0, 'rec': 0, 'table': 0, 'all': 0, 'match': 0}
        start = time.time()
        structure_res, elapse = self._structure(copy.deepcopy(img))
        result['cell_bbox'] = structure_res[1].tolist()
        time_dict['table'] = elapse

        dt_boxes, rec_res, det_elapse, rec_elapse = self._ocr(
            copy.deepcopy(img))
        time_dict['det'] = det_elapse
        time_dict['rec'] = rec_elapse

        if return_ocr_result_in_table:
            result['boxes'] = [x.tolist() for x in dt_boxes]
            result['rec_res'] = rec_res

        tic = time.time()
        pred_html = self.match(structure_res, dt_boxes, rec_res)
        toc = time.time()
        time_dict['match'] = toc - tic
        result['html'] = pred_html
        end = time.time()
        time_dict['all'] = end - start
        return result, time_dict

    def _structure(self, img):
        structure_res, elapse = self.table_structurer(copy.deepcopy(img))
        return structure_res, elapse

    def _ocr(self, img):
        h, w = img.shape[:2]
        dt_boxes, det_elapse = self.text_detector(copy.deepcopy(img))
        dt_boxes = sorted_boxes(dt_boxes)

        r_boxes = []
        for box in dt_boxes:
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        # logger.debug("dt_boxes num : {}, elapse : {}".format(
            # len(dt_boxes), det_elapse))
        if dt_boxes is None:
            return None, None

        img_crop_list = []
        for i in range(len(dt_boxes)):
            det_box = dt_boxes[i]
            x0, y0, x1, y1 = expand(2, det_box, img.shape)
            text_rect = img[int(y0):int(y1), int(x0):int(x1), :]
            img_crop_list.append(text_rect)
        rec_res, rec_elapse = self.text_recognizer(img_crop_list)
        # logger.debug("rec_res num  : {}, elapse : {}".format(
            # len(rec_res), rec_elapse))
        return dt_boxes, rec_res, det_elapse, rec_elapse



class table:
    def __init__(self, config):

        session = OrtInferSession(config)
        self.predictor, self.input_tensor, self.output_tensors, self.config = session, session.session.get_inputs()[0], None, None

        self.args = config
        
        resize_op = {'ResizeTableImage': {'max_len': config['table_max_len'], }}
        pad_op = {
            'PaddingTableImage': {
                'size': [config['table_max_len'], config['table_max_len']]
            }
        }
        normalize_op = {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225] if
                config['table_algorithm'] not in ['TableMaster'] else [0.5, 0.5, 0.5],
                'mean': [0.485, 0.456, 0.406] if
                config['table_algorithm'] not in ['TableMaster'] else [0.5, 0.5, 0.5],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }
        to_chw_op = {'ToCHWImage': None}
        keep_keys_op = {'KeepKeys': {'keep_keys': ['image', 'shape']}}
        if config['table_algorithm'] not in ['TableMaster']:
            pre_process_list = [
                resize_op, normalize_op, pad_op, to_chw_op, keep_keys_op
            ]
        else:
            pre_process_list = [
                resize_op, pad_op, normalize_op, to_chw_op, keep_keys_op
            ]

        postprocess_params = {
                'name': 'TableLabelDecode',
                "character_dict_path": config['table_char_dict_path'],
                'merge_no_span_structure': config['merge_no_span_structure']
            }

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

    def __call__(self, img):
        starttime = time.time()
        # if self.args.benchmark:
        #     self.autolog.times.start()

        # ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()
    
        outputs = self.predictor(img)

        preds = {}
        preds['structure_probs'] = outputs[1]
        preds['loc_preds'] = outputs[0]

        shape_list = np.expand_dims(data[-1], axis=0)
        post_result = self.postprocess_op(preds, [shape_list])

        structure_str_list = post_result['structure_batch_list'][0]
        bbox_list = post_result['bbox_batch_list'][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = [
            '<html>', '<body>', '<table>'
        ] + structure_str_list + ['</table>', '</body>', '</html>']
        elapse = time.time() - starttime
        # if self.args.benchmark:
        #     self.autolog.times.end(stamp=True)
        return (structure_str_list, bbox_list), elapse