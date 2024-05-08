import argparse
import copy
import json
import os
import time
import cv2
import fitz
import fastdeploy as fd
from multiprocessing import Pool
from threading import Thread
import pickle

import numpy as np

from copy import deepcopy
from latex.latex_rec import Latex2Text, save_layout_img, sort_boxes
from table.utils import TableMatch, sorted_boxes
from utils import convert_info_docx, convert_info_md, download_and_extract_models, read_yaml, sorted_layout_boxes, document_to_xl
from table.table_det import table
from PIL import Image

from cnstd import LayoutAnalyzer


device = "gpu"
device_id = 0
backend = 'default'

def load_model(layout_path, num_class, det_path, rec_path, rec_bs, formula_path, anay_path):
    # load layout model
    layout_model_file = os.path.join(layout_path, "model.pdmodel")
    layout_params_file = os.path.join(layout_path, "model.pdiparams")

    layout_option = fd.RuntimeOption()
    layout_option.use_gpu(device_id)
    layout_option.set_cpu_thread_num(10)

    global layout_model
    layout_model = fd.vision.ocr.StructureV2Layout(
        layout_model_file, layout_params_file, layout_option)

    layout_model.postprocessor.num_class = num_class
    
    global layout_labels
    layout_labels = ["text", "title", "list", "table", "figure"]
    if num_class == 10:
        layout_labels = [
            "text", "title", "figure", "figure_caption", "table", "table_caption",
            "header", "footer", "reference", "equation"
        ]

    # load det & rec model
    det_model_file = os.path.join(det_path, "inference.pdmodel")
    det_params_file = os.path.join(det_path, "inference.pdiparams")

    model_option = fd.RuntimeOption()

    if device.lower() == "gpu":
        model_option.use_gpu(device_id)
    
    if backend.lower() == "trt":
        model_option.use_trt_backend()

        model_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640], [1, 3, 960, 960])
        model_option.set_trt_cache_file(det_path + "/det_trt_cache.trt")
    
    elif backend.lower() == "pptrt":
        model_option.use_paddle_infer_backend()
        model_option.paddle_infer_option.collect_trt_shape = True
        model_option.paddle_infer_option.enable_trt = True

        model_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640], [1, 3, 960, 960])
        model_option.set_trt_cache_file(det_path)

    elif backend.lower() == "ort":
        model_option.use_ort_backend()

    elif backend.lower() == "paddle":
        model_option.use_paddle_infer_backend()

    global det_model
    det_model = fd.vision.ocr.DBDetector(det_model_file, det_params_file, runtime_option=model_option)

    rec_model_file = os.path.join(rec_path, "inference.pdmodel")
    rec_params_file = os.path.join(rec_path, "inference.pdiparams")
    rec_label_file = os.path.join(rec_path, "ppocr_keys_v1.txt")

    model_option = fd.RuntimeOption()

    if device.lower() == "gpu":
        model_option.use_gpu(device_id)
    
    if backend.lower() == "trt":
        model_option.use_tensorrt()

        model_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                        [rec_bs, 3, 48, 320],
                                        [rec_bs, 3, 48, 2304])
        model_option.set_trt_cache_file(rec_path + "/rec_trt_cache.trt")
    
    elif backend.lower() == "pptrt":
        model_option.use_paddle_infer_backend()
        model_option.paddle_infer_option.collect_trt_shape = True
        model_option.paddle_infer_option.enable_trt = True

        model_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                        [rec_bs, 3, 48, 320],
                                        [rec_bs, 3, 48, 2304])
        model_option.set_trt_cache_file(rec_path)

    elif backend.lower() == "ort":
        model_option.use_ort_backend()

    elif backend.lower() == "paddle":
        model_option.use_paddle_infer_backend()

    global rec_model
    rec_model = fd.vision.ocr.Recognizer(
        rec_model_file, rec_params_file, rec_label_file, runtime_option=model_option)
    
    det_model.preprocessor.max_side_len = 960
    det_model.postprocessor.det_db_thresh = 0.3
    det_model.postprocessor.det_db_box_thresh = 0.6
    det_model.postprocessor.det_db_unclip_ratio = 1.5
    det_model.postprocessor.det_db_score_mode = "slow"
    det_model.postprocessor.use_dilation = False
    
    global ppocr
    if "PP-OCRV4" in rec_path.upper():
        ppocr = fd.vision.ocr.PPOCRv4(
            det_model=det_model, cls_model=None, rec_model=rec_model)
        ppocr.rec_batch_size = rec_bs
    else:
        ppocr = fd.vision.ocr.PPOCRv3(
            det_model=det_model, cls_model=None, rec_model=rec_model)
        ppocr.rec_batch_size = rec_bs

    # load table model
    config = read_yaml(r'config.yaml')
    global table_structurer
    table_structurer = table(config['Table'])

    global Match
    Match = TableMatch(filter_ocr_result=True)

    global Analyzer
    Analyzer = Latex2Text(formula_config = {'model_fp': formula_path}, analyzer_config=dict(model_name='mfd', model_type='yolov7', model_fp=anay_path), device = 'gpu')


def text_system(img, thre = 0.5):
    time1 = time.time()
    result = ppocr.predict(img)
    bbox = []
    rec = []
    for i in range(len(result.rec_scores)):
        if result.rec_scores[i] > thre:
            bbox.append(np.reshape(result.boxes[i], (4, 2)))
            rec.append((result.text[i],result.rec_scores[i]))
    return bbox, rec, time.time() - time1

def expand(pix, det_box, shape):
    x0, y0, x1, y1 = det_box
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

def to_excel(html_table, excel_path):
    document_to_xl(html_table, excel_path)

def readImage(image_file) -> list:
    if os.path.basename(image_file)[-3:] == 'pdf':
        imgs = []
        with fitz.open(image_file) as pdf:
            for pg in range(0, pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
    else:
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if img is not None:
            imgs = [img]

    return imgs

def save_structure_res(res, save_folder, img_name, img_idx=0):
    excel_save_folder = os.path.join(save_folder, img_name)
    os.makedirs(excel_save_folder, exist_ok=True)
    res_cp = deepcopy(res)
    # save res
    with open(
            os.path.join(excel_save_folder, 'res_{}.txt'.format(img_idx)),
            'w',
            encoding='utf8') as f:
        for region in res_cp:
            roi_img = region.pop('img')
            try:
                f.write('{}\n'.format(json.dumps(region)))
            except Exception as e:
                print(e)
                print(region)

            if region['type'].lower() == 'table' and len(region[
                    'res']) > 0 and 'html' in region['res']:
                if region['res']['html'] is None:
                    img_path = os.path.join(
                        excel_save_folder,
                        '{}_{}.jpg'.format(region['bbox'], img_idx))
                    cv2.imwrite(img_path, roi_img)
                else:
                    excel_path = os.path.join(
                        excel_save_folder,
                        '{}_{}.xlsx'.format(region['bbox'], img_idx))
                    to_excel(region['res']['html'], excel_path)
            elif region['type'].lower() == 'figure':
                img_path = os.path.join(
                    excel_save_folder,
                    '{}_{}.jpg'.format(region['bbox'], img_idx))
                cv2.imwrite(img_path, roi_img)

def process_predict(pdf_info, img_idx=0):
    images = readImage(pdf_info[0])
    all_res = []
    start = time.time()
    for image in images:
        ori_im = image.copy()
        if layout_model is not None:
            layout_time1 = time.time()
            result = layout_model.predict(ori_im)
            print('layout time:', time.time() - layout_time1)
            layout_res = []
            for i in range(len(result.label_ids)):
                layout_res.append({'bbox': np.asarray(result.boxes[i]), 'label': layout_labels[result.label_ids[i]]})
        else:
            h, w = ori_im.shape[:2]
            layout_res = [dict(bbox=None, label='table')]
        res_list = []
        for region in layout_res:
            res = ''
            if region['bbox'] is not None:
                x1, y1, x2, y2 = region['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                roi_img = ori_im[y1:y2, x1:x2, :]
            else:
                x1, y1, x2, y2 = 0, 0, w, h
                roi_img = ori_im
            if region['label'] == 'table':
                table_time1 = time.time()

                res = dict()
                structure_res, elapse = table_structurer(roi_img.copy())
                res['cell_bbox'] = structure_res[1].tolist()
                h, w = roi_img.shape[:2]
                dt_boxes, rec_res, ocr_time_dict = text_system(roi_img.copy())
                r_boxes = []
                for box in dt_boxes:
                    x_min = max(0, box[:, 0].min() - 1)
                    x_max = min(w, box[:, 0].max() + 1)
                    y_min = max(0, box[:, 1].min() - 1)
                    y_max = min(h, box[:, 1].max() + 1)
                    box = [x_min, y_min, x_max, y_max]
                    r_boxes.append(box)
                dt_boxes = np.array(r_boxes)
                if dt_boxes is None:
                    res = {'html': None}
                    region['label'] = 'figure'
                res['html'] = Match(structure_res, dt_boxes, rec_res)
                print('table time:', time.time() - table_time1)

            else:
                
                rec_time1 = time.time()
                wht_im = np.ones(ori_im.shape, dtype=ori_im.dtype)
                wht_im[y1:y2, x1:x2, :] = roi_img

                lax_img, mf_out = Analyzer.recognize_by_cnstd(wht_im, resized_shape=608)

                if mf_out == None:

                    filter_boxes, filter_rec_res, ocr_time_dict = text_system(wht_im)
                    style_token = [
                        '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                        '</b>', '<sub>', '</sup>', '<overline>',
                        '</overline>', '<underline>', '</underline>', '<i>',
                        '</i>'
                    ]
                    res = []
                    for box, rec_res in zip(filter_boxes, filter_rec_res):
                        rec_str, rec_conf = rec_res
                        for token in style_token:
                            if token in rec_str:
                                rec_str = rec_str.replace(token, '')
                        box += [x1, y1]
                        res.append({
                            'text': rec_str,
                            'confidence': float(rec_conf),
                            'text_region': box.tolist()
                        })
                    print('rec time:', time.time() - rec_time1)
                
                else:
                    
                    lax_img = np.array(lax_img)
                    filter_boxes, filter_rec_res, ocr_time_dict = text_system(lax_img)
                    style_token = [
                        '<strike>', '<strike>', '<sup>', '</sub>', '<b>',
                        '</b>', '<sub>', '</sup>', '<overline>',
                        '</overline>', '<underline>', '</underline>', '<i>',
                        '</i>'
                    ]
                    st_res =  []
                    for lat in mf_out:
                        lat['position']+=[x1,y1]
                        st_res.append({
                            'text': lat['text'],
                            'confidence': float(0.8),
                            'position': lat['position']#.tolist()
                            })
                    for box, rec_res in zip(filter_boxes, filter_rec_res):
                        rec_str, rec_conf = rec_res
                        for token in style_token:
                            if token in rec_str:
                                rec_str = rec_str.replace(token, '')
                        box += [x1, y1]
                        st_res.append({
                            'text': rec_str,
                            'confidence': float(rec_conf),
                            'position': box#.tolist()
                        })

                    st_res = sort_boxes(st_res, key='position')

                    res = []
                    for i in st_res:
                        i = i[0]
                        i['text_region'] = i['position'].tolist()
                        del i['position']
                        res.append(i)
                    
                    print('rec time:', time.time() - rec_time1)

            res_list.append({
                'type': region['label'].lower(),
                'bbox': [x1, y1, x2, y2],
                'img': roi_img,
                'res': res,
                'img_idx': img_idx
            })
        end = time.time()
        # print(end - start)

        save_structure_res(res_list, "output", pdf_info[1])
        h, w, _ = image.shape
        res = sorted_layout_boxes(res_list, w)
        all_res += res
    convert_info_md(images, all_res, "output", pdf_info[1])
    # save all_res to json
    with open(os.path.join(os.path.join("output", pdf_info[1]), 'res.pkl'), "wb") as f:
        pickle.dump({'data': all_res, 'name': pdf_info[0]}, f)

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', help='Path to the folder containing PDF files.')
    parser.add_argument('--use-multi-process', action='store_true', help='Whether to use multi-processing.')
    parser.add_argument('--process-num', type=int, default=4, help='Number of processes to use.')
    parser.add_argument('--layout-model-path', default='models/picodet_lcnet_x1_0_fgd_layout_cdla_infer', help='Path to the layout model.')
    parser.add_argument('--det-path', default='models/ch_PP-OCRv4_det_infer', help='Path to the detection model.')
    parser.add_argument('--rec-path', default='models/ch_PP-OCRv4_rec_infer', help='Path to the recognition model.')
    parser.add_argument('--formula-path', default='models/latex_rec.pth', help='Path to the LaTeX recognition model.')
    parser.add_argument('--anay-path', default='models/mfd.pt', help='Path to the analysis model.')
    parser.add_argument('--rec-bs', type=int, default=16, help='Recognition batch size.')
    args = parser.parse_args()

    download_and_extract_models()

    use_multi_process = args.use_multi_process
    process_num = args.process_num
    root_path = args.root_path
    file_names = os.listdir(root_path)

    layout_model_path = args.layout_model_path
    det_path = args.det_path
    rec_path = args.rec_path

    num_class = 10 if "cdla" in layout_model_path.lower() else 5

    file_info = [(os.path.join(root_path, name), name) for name in file_names]

    with Pool(
            process_num,
            initializer=load_model,
            initargs=(layout_model_path, num_class, det_path, rec_path, 
                      args.rec_bs, args.formula_path, args.anay_path)) as pool:
        time1 = time.time()
        pool.map(process_predict, file_info)
        print('------------all time:', time.time() - time1, '------------')