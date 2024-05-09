import os
from glob import glob
import logging
from itertools import chain
from pathlib import Path
import random
import re
from typing import Dict, Any, Optional, Union, List
from copy import deepcopy, copy

from PIL import Image, ImageOps
import numpy as np
import torch

from latex.base_model import LatexOCR
# utils
import platform
from functools import cmp_to_key


from cnstd.utils import get_model_file
from cnstd import LayoutAnalyzer
from cnstd.yolov7.consts import CATEGORY_DICT
from cnstd.yolov7.general import xyxy24p, box_partial_overlap

def data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'pix2text')
    else:
        return os.path.join(os.path.expanduser("~"), '.pix2text')

def data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('PIX2TEXT_HOME', data_dir_default())

def read_img(
    path: Union[str, Path], return_type='Tensor'
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """

    Args:
        path (str): image file path
        return_type (str): 返回类型；
            支持 `Tensor`，返回 torch.Tensor；`ndarray`，返回 np.ndarray；`Image`，返回 `Image.Image`

    Returns: RGB Image.Image, or np.ndarray / torch.Tensor, with shape [Channel, Height, Width]
    """
    assert return_type in ('Tensor', 'ndarray', 'Image')
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert('RGB')  # 识别旋转后的图片（pillow不会自动识别）
    if return_type == 'Image':
        return img
    img = np.array(img)
    if return_type == 'ndarray':
        return img
    return torch.tensor(img.transpose((2, 0, 1)))

from torchvision.utils import save_image
def save_img(img: Union[torch.Tensor, np.ndarray], path):
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    # img *= 255
    # img = img.to(dtype=torch.uint8)
    save_image(img, path)

    # Image.fromarray(img).save(path)


logger = logging.getLogger(__name__)

# 图片分类模型对应的类别标签
IMAGE_TYPES = ('general', 'english', 'formula')

# LATEX_OCR 使用的配置信息
LATEX_CONFIG_FP = Path(__file__).parent.absolute() / 'latex_config.yaml'


CLF_MODEL_URL_FMT = '%s.zip'

DEFAULT_CONFIGS = {
    'analyzer': {'model_name': 'mfd'},
    'clf': {
        'base_model_name': 'mobilenet_v2',
        'categories': IMAGE_TYPES,
        'transform_configs': {
            'crop_size': [150, 450],
            'resize_size': 160,
            'resize_max_size': 1000,
        },
        'model_dir': Path(data_dir()) / 'clf',
        'model_fp': None,  # 如果指定，直接使用此模型文件
    },
    'general': {},
    'english': {'det_model_name': 'en_PP-OCRv3_det', 'rec_model_name': 'en_PP-OCRv3'},
    'formula': {
        'config': LATEX_CONFIG_FP,
        'checkpoint': Path(data_dir()) / 'formula' / 'weights.pth',
        'no_resize': False,
    },
    'thresholds': {  # 用于clf场景
        'formula2general': 0.65,  # 如果识别为 `formula` 类型，但阈值小于此值，则改为 `general` 类型
        'english2general': 0.75,  # 如果识别为 `english` 类型，但阈值小于此值，则改为 `general` 类型
    },
}

def merge_line_texts(
    out: List[Dict[str, Any]], auto_line_break: bool = True, line_sep='\n'
) -> str:
    """
    把 Pix2Text.recognize_by_mfd() 的返回结果，合并成单个字符串
    Args:
        out (List[Dict[str, Any]]):
        auto_line_break: 基于box位置自动判断是否该换行

    Returns: 合并后的字符串

    """
    out_texts = []
    line_margin_list = []  # 每行的最左边和左右边的x坐标
    isolated_included = []  # 每行是否包含了 `isolated` 类型的数学公式
    for o in out:
        line_number = o.get('line_number', 0)
        if len(out_texts) <= line_number:
            out_texts.append([])
            line_margin_list.append([0, 0])
            isolated_included.append(False)
        out_texts[line_number].append(o['text'])
        line_margin_list[line_number][1] = max(
            line_margin_list[line_number][1], float(o['position'][2, 0])
        )
        line_margin_list[line_number][0] = min(
            line_margin_list[line_number][0], float(o['position'][0, 0])
        )
        if o['type'] == 'isolated':
            isolated_included[line_number] = True

    line_text_list = [smart_join(o) for o in out_texts]

    if not auto_line_break:
        return line_sep.join(line_text_list)

    line_lengths = [rx - lx for lx, rx in line_margin_list]
    line_length_thrsh = max(line_lengths) * 0.3

    lines = np.array(
        [
            margin
            for idx, margin in enumerate(line_margin_list)
            if isolated_included[idx] or line_lengths[idx] >= line_length_thrsh
        ]
    )
    min_x, max_x = lines.max(axis=0)

    indentation_thrsh = (max_x - min_x) * 0.1
    res_line_texts = [''] * len(line_text_list)
    for idx, txt in enumerate(line_text_list):
        if isolated_included[idx]:
            res_line_texts[idx] = line_sep + txt + line_sep
            continue

        tmp = txt
        if line_margin_list[idx][0] > min_x + indentation_thrsh:
            tmp = line_sep + txt
        if line_margin_list[idx][1] < max_x - indentation_thrsh:
            tmp = tmp + line_sep
        res_line_texts[idx] = tmp

    out = smart_join(res_line_texts)
    return out.replace(line_sep + line_sep, line_sep)  # 把 '\n\n' 替换为 '\n'

def list2box(xmin, ymin, xmax, ymax):
    return np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
    )

COLOR_LIST = [
    [0, 140, 255],  # 深橙色
    [127, 255, 0],  # 春绿色
    [255, 144, 30],  # 道奇蓝
    [180, 105, 255],  # 粉红色
    [128, 0, 128],  # 紫色
    [0, 255, 255],  # 黄色
    [255, 191, 0],  # 深天蓝色
    [50, 205, 50],  # 石灰绿色
    [60, 20, 220],  # 猩红色
    [130, 0, 75]  # 靛蓝色
]
def save_layout_img(img0, categories, one_out, save_path, key='position'):
    import cv2
    from cnstd.yolov7.plots import plot_one_box

    """可视化版面分析结果。"""
    if isinstance(img0, Image.Image):
        img0 = cv2.cvtColor(np.asarray(img0.convert('RGB')), cv2.COLOR_RGB2BGR)

    if len(categories) > 10:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in categories]
    else:
        colors = COLOR_LIST
    for one_box in one_out:
        _type = one_box['type']
        box = one_box[key]
        xyxy = [box[0, 0], box[0, 1], box[2, 0], box[2, 1]]
        label = f'{_type}'
        plot_one_box(
            xyxy,
            img0,
            label=label,
            color=colors[categories.index(_type)],
            line_thickness=1,
        )

    cv2.imwrite(save_path, img0)
    logger.info(f" The image with the result is saved in: {save_path}")

def rotated_box_to_horizontal(box):
    """将旋转框转换为水平矩形。

    :param box: [4, 2]，左上角、右上角、右下角、左下角的坐标
    """
    xmin = min(box[:, 0])
    xmax = max(box[:, 0])
    ymin = min(box[:, 1])
    ymax = max(box[:, 1])
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

def is_chinese(ch):
    """
    判断一个字符是否为中文字符
    """
    return '\u4e00' <= ch <= '\u9fff'
def smart_join(str_list):
    """
    对字符串列表进行拼接，如果相邻的两个字符串都是中文或包含空白符号，则不加空格；其他情况则加空格
    """

    def contain_whitespace(s):
        if re.search(r'\s', s):
            return True
        else:
            return False

    res = str_list[0]
    for i in range(1, len(str_list)):
        if (is_chinese(res[-1]) and is_chinese(str_list[i][0])) or contain_whitespace(
            res[-1] + str_list[i][0]
        ):
            res += str_list[i]
        else:
            res += ' ' + str_list[i]
    return res

def is_valid_box(box, min_height=8, min_width=2) -> bool:
    """判断box是否有效。
    :param box: [4, 2]，左上角、右上角、右下角、左下角的坐标
    :param min_height: 最小高度
    :param min_width: 最小宽度
    :return: bool, 是否有效
    """
    return (
        box[0, 0] + min_width <= box[1, 0]
        and box[1, 1] + min_height <= box[2, 1]
        and box[2, 0] >= box[3, 0] + min_width
        and box[3, 1] >= box[0, 1] + min_height
    )

def overlap(box1, box2, key='position'):
    # 计算它们在y轴上的IOU: Interaction / min(height1, height2)
    # 判断是否有交集
    box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
    box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]
    if box1[3] <= box2[1] or box2[3] <= box1[1]:
        return 0
    # 计算交集的高度
    y_min = max(box1[1], box2[1])
    y_max = min(box1[3], box2[3])
    return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))

def _compare_box(box1, box2, anchor, key, left_best: bool = True):
    over1 = overlap(box1, anchor, key)
    over2 = overlap(box2, anchor, key)
    if box1[key][2, 0] < box2[key][0, 0] - 3:
        return -1
    elif box2[key][2, 0] < box1[key][0, 0] - 3:
        return 1
    else:
        if max(over1, over2) >= 3 * min(over1, over2):
            return over2 - over1 if left_best else over1 - over2
        return box1[key][0, 0] - box2[key][0, 0]

def sort_and_filter_line_boxes(line_boxes, key):
    if len(line_boxes) <= 1:
        return line_boxes

    allowed_max_overlay_x = 20

    def find_right_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(
            max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95
        )
        right_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box['line_number'] < 0
            and l_box[key][0, 0] >= anchor[key][2, 0] - allowed_max
        ]
        if not right_boxes:
            return None
        right_boxes = sorted(
            right_boxes,
            key=cmp_to_key(
                lambda x, y: _compare_box(x, y, anchor, key, left_best=True)
            ),
        )
        return right_boxes[0]

    def find_left_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(
            max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95
        )
        left_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box['line_number'] < 0
            and l_box[key][2, 0] <= anchor[key][0, 0] + allowed_max
        ]
        if not left_boxes:
            return None
        left_boxes = sorted(
            left_boxes,
            key=cmp_to_key(
                lambda x, y: _compare_box(x, y, anchor, key, left_best=False)
            ),
        )
        return left_boxes[-1]

    res_boxes = [line_boxes[0]]
    anchor = res_boxes[0]
    line_number = anchor['line_number']

    while True:
        right_box = find_right_box(anchor)
        if right_box is None:
            break
        right_box['line_number'] = line_number
        res_boxes.append(right_box)
        anchor = right_box

    anchor = res_boxes[0]
    while True:
        left_box = find_left_box(anchor)
        if left_box is None:
            break
        left_box['line_number'] = line_number
        res_boxes.insert(0, left_box)
        anchor = left_box

    return res_boxes

def get_same_line_boxes(anchor, total_boxes):
    line_boxes = [anchor]
    for box in total_boxes:
        if box['line_number'] >= 0:
            continue
        if max([overlap(box, l_box) for l_box in line_boxes]) > 0.1:
            line_boxes.append(box)
    return line_boxes

def sort_boxes(boxes: List[dict], key='position') -> List[List[dict]]:
    # 按y坐标排序所有的框
    boxes.sort(key=lambda box: box[key][0, 1])
    for box in boxes:
        box['line_number'] = -1  # 所在行号，-1表示未分配

    def get_anchor():
        anchor = None
        for box in boxes:
            if box['line_number'] == -1:
                anchor = box
                break
        return anchor

    lines = []
    while True:
        anchor = get_anchor()
        if anchor is None:
            break
        anchor['line_number'] = len(lines)
        line_boxes = get_same_line_boxes(anchor, boxes)
        line_boxes = sort_and_filter_line_boxes(line_boxes, key)
        lines.append(line_boxes)

    return lines


def xyxy24p(x, ret_type=torch.Tensor):
    xmin, ymin, xmax, ymax = [float(_x) for _x in x]
    out = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
    if ret_type is not None:
        return ret_type(out).reshape((4, 2))
    return out

class Latex2Text(object):
    MODEL_FILE_PREFIX = 'pix2text-v{}'.format('0.0.1')

    def __init__(
        self,
        *,
        analyzer_config: Dict[str, Any] = None,
        clf_config: Dict[str, Any] = None,
        general_config: Dict[str, Any] = None,
        english_config: Dict[str, Any] = None,
        formula_config: Dict[str, Any] = None,
        thresholds: Dict[str, Any] = None,
        device: str = 'cpu',  # ['cpu', 'cuda', 'gpu']
        **kwargs,
    ):
        """

        Args:
            analyzer_config (dict): Analyzer模型对应的配置信息；默认为 `None`，表示使用默认配置
            clf_config (dict): 分类模型对应的配置信息；默认为 `None`，表示使用默认配置
            general_config (dict): 通用模型对应的配置信息；默认为 `None`，表示使用默认配置
            english_config (dict): 英文模型对应的配置信息；默认为 `None`，表示使用默认配置
            formula_config (dict): 公式识别模型对应的配置信息；默认为 `None`，表示使用默认配置
            thresholds (dict): 识别阈值对应的配置信息；默认为 `None`，表示使用默认配置
            device (str): 使用什么资源进行计算，支持 `['cpu', 'cuda', 'gpu']`；默认为 `cpu`
            **kwargs (): 预留的其他参数；目前未被使用
        """
        if device.lower() == 'gpu':
            device = 'cuda'
        self.device = device
        thresholds = thresholds or DEFAULT_CONFIGS['thresholds']
        self.thresholds = deepcopy(thresholds)

        (
            analyzer_config,
            clf_config,
            general_config,
            english_config,
            formula_config,
        ) = self._prepare_configs(
            analyzer_config,
            clf_config,
            general_config,
            english_config,
            formula_config,
            device,
        )

        self.analyzer = LayoutAnalyzer(**analyzer_config)

        _clf_config = deepcopy(clf_config)
        _clf_config.pop('model_dir')
        _clf_config.pop('model_fp')
        # self.image_clf = ImageClassifier(**_clf_config)

        # self.general_ocr = CnOcr(**general_config)
        # self.english_ocr = CnOcr(**english_config)
        self.latex_model = LatexOCR(formula_config)

        # self._assert_and_prepare_clf_model(clf_config)

    def _prepare_configs(
        self,
        analyzer_config,
        clf_config,
        general_config,
        english_config,
        formula_config,
        device,
    ):
        def _to_default(_conf, _def_val):
            if not _conf:
                _conf = _def_val
            return _conf

        analyzer_config = _to_default(analyzer_config, DEFAULT_CONFIGS['analyzer'])
        analyzer_config['device'] = device
        clf_config = _to_default(clf_config, DEFAULT_CONFIGS['clf'])
        general_config = _to_default(general_config, DEFAULT_CONFIGS['general'])
        general_config['context'] = device
        english_config = _to_default(english_config, DEFAULT_CONFIGS['english'])
        english_config['context'] = device
        formula_config = _to_default(formula_config, DEFAULT_CONFIGS['formula'])
        formula_config['device'] = device
        return (
            analyzer_config,
            clf_config,
            general_config,
            english_config,
            formula_config,
        )

    def _assert_and_prepare_clf_model(self, clf_config):
        model_file_prefix = '{}-{}'.format(
            self.MODEL_FILE_PREFIX, clf_config['base_model_name']
        )
        model_dir = clf_config['model_dir']
        model_fp = clf_config['model_fp']

        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        fps = glob(os.path.join(model_dir, model_file_prefix) + '*.ckpt')
        if len(fps) > 1:
            raise ValueError(
                'multiple .ckpt files are found in %s, not sure which one should be used'
                % model_dir
            )
        elif len(fps) < 1:
            logger.warning('no .ckpt file is found in %s' % model_dir)
            # url = format_hf_hub_url(CLF_MODEL_URL_FMT % clf_config['base_model_name'])
            # get_model_file(url, model_dir)  # download the .zip file and unzip
            fps = glob(os.path.join(model_dir, model_file_prefix) + '*.ckpt')

        model_fp = fps[0]
        self.image_clf.load(model_fp, self.device)

    @classmethod
    def from_config(cls, total_configs: Optional[dict] = None, device: str = 'cpu'):
        total_configs = total_configs or DEFAULT_CONFIGS
        return cls(
            analyzer_config=total_configs.get('analyzer', dict()),
            clf_config=total_configs.get('clf', dict()),
            general_config=total_configs.get('general', dict()),
            english_config=total_configs.get('english', dict()),
            formula_config=total_configs.get('formula', dict()),
            thresholds=total_configs.get('thresholds', DEFAULT_CONFIGS['thresholds']),
            device=device,
        )

    def __call__(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        return self.recognize(img, **kwargs)

    def recognize(
        self, img: Union[str, Path, Image.Image], use_analyzer: bool = True, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        对图片先做版面分析，然后再识别每块中包含的信息。在版面分析未识别出内容时，则把整个图片作为整体进行识别。

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            use_analyzer (bool): whether to use the analyzer (MFD or Layout) to analyze the image; Default: `True`
            kwargs ():
                * resized_shape (int): 把图片宽度resize到此大小再进行处理；默认值为 `700`
                * save_analysis_res (str): 把解析结果图片存在此文件中；默认值为 `None`，表示不存储
                * embed_sep (tuple): embedding latex的前后缀；只针对使用 `MFD` 时才有效；默认值为 `(' $', '$ ')`
                * isolated_sep (tuple): isolated latex的前后缀；只针对使用 `MFD` 时才有效；默认值为 `('$$\n', '\n$$')`

        Returns: a list of dicts, with keys:
           `type`: 图像类别；
           `text`: 识别出的文字或Latex公式
           `postion`: 所在块的位置信息，`np.ndarray`, with shape of [4, 2]

        """
        out = None
        # if use_analyzer:
        #     if self.analyzer._model_name == 'mfd':
        #         out = self.recognize_by_mfd(img, **kwargs)
        #     else:
        #         out = self.recognize_by_layout(img, **kwargs)
        if not out:
            out = self.recognize_by_clf(img, **kwargs)
        return out

    def recognize_by_clf(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        把整张图片作为一整块进行识别。

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`

        Returns: a list of dicts, with keys:
           `type`: 图像类别；
           `text`: 识别出的文字或Latex公式
           `position`: 所在块的位置信息，`np.ndarray`, with shape of [4, 2]

        """
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')
        width, height = img0.size
        # _img = torch.tensor(np.asarray(img0))
        # res = 'formula'
        # logger.debug('CLF Result: %s', res)

        # image_type = res[0]
        # if res[1] < self.thresholds['formula2general'] and res[0] == 'formula':
        #     image_type = 'general'
        # if res[1] < self.thresholds['english2general'] and res[0] == 'english':
        #     image_type = 'general'
        # if image_type == 'formula':
        #     result = self._latex(img)
        # else:
        #     result = self._ocr(img, image_type)

        result = self._latex(img)

        box = xyxy24p([0, 0, width, height], np.array)

        # if kwargs.get('save_analysis_res'):
        #     out = [{'type': image_type, 'score': res[1], 'position': box}]
        #     save_layout_img(img0, IMAGE_TYPES, out, kwargs.get('save_analysis_res'))

        return [{'text': result, 'position': box}]
    
    def recognize_by_cnstd(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        对图片先做MFD 或 版面分析，然后再识别每块中包含的信息。

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            kwargs ():
                * resized_shape (int): 把图片宽度resize到此大小再进行处理；默认值为 `608`
                * save_analysis_res (str): 把解析结果图片存在此文件中；默认值为 `None`，表示不存储
                * embed_sep (tuple): embedding latex的前后缀；默认值为 `(' $', '$ ')`
                * isolated_sep (tuple): isolated latex的前后缀；默认值为 `('$$\n', '\n$$')`

        Returns: a list of ordered (top to bottom, left to right) dicts,
            with each dict representing one detected box, containing keys:
           `type`: 图像类别；Optional: 'text', 'isolated', 'embedding'
           `text`: 识别出的文字或Latex公式
           `position`: 所在块的位置信息，`np.ndarray`, with shape of [4, 2]
           `line_number`: box 所在行号（第一行 `line_number==0`），值相同的box表示它们在同一行

        """
        resized_shape = kwargs.get('resized_shape', 608)
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        elif isinstance(img, np.ndarray):
            img0 = Image.fromarray(img)
        else:
            img0 = read_img(img, return_type='Image')
        w, h = img0.size
        ratio = resized_shape / w
        resized_shape = (int(h * ratio), resized_shape)  # (H, W)
        analyzer_outs = self.analyzer(img0.copy(), resized_shape=resized_shape, conf_threshold = 0.65)

        if not analyzer_outs:
            return None, None

        logger.debug('MFD Result: %s', analyzer_outs)
        embed_sep = kwargs.get('embed_sep', (' $', '$ '))
        isolated_sep = kwargs.get('isolated_sep', ('$$\n', '\n$$'))

        mf_out = []
        for box_info in analyzer_outs:
            box = box_info['box']
            xmin, ymin, xmax, ymax = (
                int(box[0][0]),
                int(box[0][1]),
                int(box[2][0]),
                int(box[2][1]),
            )
            crop_patch = img0.crop((xmin, ymin, xmax, ymax))
            patch_out = self._latex(crop_patch)
            sep = isolated_sep if box_info['type'] == 'isolated' else embed_sep
            text = sep[0] + patch_out + sep[1]
            mf_out.append({'type': box_info['type'], 'text': text, 'position': box})

        img = np.array(img0.copy())
        # 把公式部分mask掉，然后对其他部分进行OCR
        for box_info in analyzer_outs:
            if box_info['type'] in ('isolated', 'embedding'):
                box = box_info['box']
                xmin, ymin = max(0, int(box[0][0]) - 1), max(0, int(box[0][1]) - 1)
                xmax, ymax = (
                    min(img0.size[0], int(box[2][0]) + 1),
                    min(img0.size[1], int(box[2][1]) + 1),
                )
                img[ymin:ymax, xmin:xmax, :] = 255

        return  img, mf_out

    def recognize_by_mfd(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        对图片先做MFD 或 版面分析，然后再识别每块中包含的信息。

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            kwargs ():
                * resized_shape (int): 把图片宽度resize到此大小再进行处理；默认值为 `608`
                * save_analysis_res (str): 把解析结果图片存在此文件中；默认值为 `None`，表示不存储
                * embed_sep (tuple): embedding latex的前后缀；默认值为 `(' $', '$ ')`
                * isolated_sep (tuple): isolated latex的前后缀；默认值为 `('$$\n', '\n$$')`

        Returns: a list of ordered (top to bottom, left to right) dicts,
            with each dict representing one detected box, containing keys:
           `type`: 图像类别；Optional: 'text', 'isolated', 'embedding'
           `text`: 识别出的文字或Latex公式
           `position`: 所在块的位置信息，`np.ndarray`, with shape of [4, 2]
           `line_number`: box 所在行号（第一行 `line_number==0`），值相同的box表示它们在同一行

        """
        # 对于大图片，把图片宽度resize到此大小；宽度比此小的图片，其实不会放大到此大小，
        # 具体参考：cnstd.yolov7.layout_analyzer.LayoutAnalyzer._preprocess_images 中的 `letterbox` 行
        resized_shape = kwargs.get('resized_shape', 608)
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')
        w, h = img0.size
        ratio = resized_shape / w
        resized_shape = (int(h * ratio), resized_shape)  # (H, W)
        analyzer_outs = self.analyzer(img0.copy(), resized_shape=resized_shape)
        logger.debug('MFD Result: %s', analyzer_outs)
        embed_sep = kwargs.get('embed_sep', (' $', '$ '))
        isolated_sep = kwargs.get('isolated_sep', ('$$\n', '\n$$'))

        mf_out = []
        for box_info in analyzer_outs:
            box = box_info['box']
            xmin, ymin, xmax, ymax = (
                int(box[0][0]),
                int(box[0][1]),
                int(box[2][0]),
                int(box[2][1]),
            )
            crop_patch = img0.crop((xmin, ymin, xmax, ymax))
            patch_out = self._latex(crop_patch)
            sep = isolated_sep if box_info['type'] == 'isolated' else embed_sep
            text = sep[0] + patch_out + sep[1]
            mf_out.append({'type': box_info['type'], 'text': text, 'position': box})

        img = np.array(img0.copy())
        # 把公式部分mask掉，然后对其他部分进行OCR
        for box_info in analyzer_outs:
            if box_info['type'] in ('isolated', 'embedding'):
                box = box_info['box']
                xmin, ymin = max(0, int(box[0][0]) - 1), max(0, int(box[0][1]) - 1)
                xmax, ymax = (
                    min(img0.size[0], int(box[2][0]) + 1),
                    min(img0.size[1], int(box[2][1]) + 1),
                )
                img[ymin:ymax, xmin:xmax, :] = 255

        box_infos = self.general_ocr.det_model.detect(img)

        def _to_iou_box(ori):
            return torch.tensor([ori[0][0], ori[0][1], ori[2][0], ori[2][1]]).unsqueeze(
                0
            )

        total_text_boxes = []
        for crop_img_info in box_infos['detected_texts']:
            # crop_img_info['box'] 可能是一个带角度的矩形框，需要转换成水平的矩形框
            hor_box = rotated_box_to_horizontal(crop_img_info['box'])
            if not is_valid_box(hor_box, min_height=8, min_width=2):
                continue
            line_box = _to_iou_box(hor_box)
            embed_mfs = []
            for box_info in mf_out:
                if box_info['type'] == 'embedding':
                    box2 = _to_iou_box(box_info['position'])
                    if float(box_partial_overlap(line_box, box2).squeeze()) > 0.7:
                        embed_mfs.append(
                            {
                                'position': box2[0].int().tolist(),
                                'text': box_info['text'],
                                'type': box_info['type'],
                            }
                        )

            ocr_boxes = self._split_line_image(line_box, embed_mfs)
            total_text_boxes.extend(ocr_boxes)

        outs = copy(mf_out)
        for box in total_text_boxes:
            crop_patch = torch.tensor(np.asarray(img0.crop(box['position'])))
            part_res = self._ocr_for_single_line(crop_patch, 'general')
            if part_res['text']:
                box['position'] = list2box(*box['position'])
                box['text'] = part_res['text']
                outs.append(box)

        outs = sort_boxes(outs, key='position')
        logger.debug(outs)
        outs = self._post_process(outs)

        outs = list(chain(*outs))
        if kwargs.get('save_analysis_res'):
            save_layout_img(
                img0,
                ('text', 'isolated', 'embedding'),
                outs,
                kwargs.get('save_analysis_res'),
            )

        return outs

    @classmethod
    def _post_process(cls, outs):
        for line_boxes in outs:
            if (
                len(line_boxes) > 1
                and line_boxes[-1]['type'] == 'text'
                and line_boxes[-2]['type'] != 'text'
            ):
                if line_boxes[-1]['text'].lower() == 'o':
                    line_boxes[-1]['text'] = '。'
        return outs

    @classmethod
    def _split_line_image(cls, line_box, embed_mfs):
        # 利用embedding formula所在位置，把单行文字图片切割成多个小段，之后这些小段会分别扔进OCR进行文字识别
        line_box = line_box[0]
        if not embed_mfs:
            return [{'position': line_box.int().tolist(), 'type': 'text'}]
        embed_mfs.sort(key=lambda x: x['position'][0])

        outs = []
        start = int(line_box[0])
        xmax, ymin, ymax = int(line_box[2]), int(line_box[1]), int(line_box[-1])
        for mf in embed_mfs:
            _xmax = min(xmax, int(mf['position'][0]) + 1)
            if start + 8 < _xmax:
                outs.append({'position': [start, ymin, _xmax, ymax], 'type': 'text'})
            start = int(mf['position'][2])
        if start < xmax:
            outs.append({'position': [start, ymin, xmax, ymax], 'type': 'text'})
        return outs

    def recognize_by_layout(
        self, img: Union[str, Path, Image.Image], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        对图片先做版面分析，然后再识别每块中包含的信息。

        Args:
            img (str or Image.Image): an image path, or `Image.Image` loaded by `Image.open()`
            kwargs ():
                * resized_shape (int): 把图片宽度resize到此大小再进行处理；默认值为 `700`
                * save_analysis_res (str): 把解析结果图片存在此文件中；默认值为 `None`，表示不存储

        Returns: a list of dicts, with keys:
           `type`: 图像类别；
           `text`: 识别出的文字或Latex公式
           `position`: 所在块的位置信息，`np.ndarray`, with shape of [4, 2]

        """
        if isinstance(img, Image.Image):
            img0 = img.convert('RGB')
        else:
            img0 = read_img(img, return_type='Image')
        resized_shape = kwargs.get('resized_shape', 500)
        layout_out = self.analyzer(img0.copy(), resized_shape=resized_shape)
        logger.debug('Layout Analysis Result: %s', layout_out)

        out = []
        for box_info in layout_out:
            box = box_info['box']
            xmin, ymin, xmax, ymax = (
                int(box[0][0]),
                int(box[0][1]),
                int(box[2][0]),
                int(box[2][1]),
            )
            crop_patch = img0.crop((xmin, ymin, xmax, ymax))
            if box_info['type'] == 'Equation':
                image_type = 'formula'
                patch_out = self._latex(crop_patch)
            else:
                crop_patch = torch.tensor(np.asarray(crop_patch))
                res = self.image_clf.predict_images([crop_patch])[0]
                image_type = res[0]
                if res[0] == 'formula':
                    image_type = 'general'
                elif (
                    res[1] < self.thresholds['english2general'] and res[0] == 'english'
                ):
                    image_type = 'general'
                patch_out = self._ocr(crop_patch, image_type)
            out.append({'type': image_type, 'text': patch_out, 'position': box})

        # if kwargs.get('save_analysis_res'):
        #     save_layout_img(
        #         img0,
        #         CATEGORY_DICT['layout'],
        #         layout_out,
        #         kwargs.get('save_analysis_res'),
        #         key='box',
        #     )

        return out

    def _ocr_for_single_line(self, image, image_type):
        ocr_model = self.english_ocr if image_type == 'english' else self.general_ocr
        try:
            return ocr_model.ocr_for_single_line(image)
        except:
            return {'text': '', 'score': 0.0}

    def _ocr(self, image, image_type):
        ocr_model = self.english_ocr if image_type == 'english' else self.general_ocr
        result = ocr_model.ocr(image)
        texts = [_one['text'] for _one in result]
        result = '\n'.join(texts)
        return result

    def _latex(self, image):
        if isinstance(image, (str, Path)):
            image = read_img(image, return_type='Image')
        out = self.latex_model(image)
        return str(out)
    


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", type=str, help="image_dir|image_path")
    # parser.add_argument("--config_path", type=str, default="config.yaml")
    # args = parser.parse_args()

    # config = read_yaml(args.config_path)
    # text_recognizer = TextRecognizer(config)

    # img = cv2.imread(args.image_path)
    # rec_res, predict_time = text_recognizer(img)
    # print(f"rec result: {rec_res}\t cost: {predict_time}s")

    l2t = Latex2Text(formula_config = {'model_fp': r'/home/onnxruntime/models/latex_rec.pth'}, analyzer_config=dict(model_name='mfd', model_type='yolov7', model_fp="/home/onnxruntime/models/mfd.pt"), device = 'gpu')

    outs = l2t.recognize_by_mfd(r'/home/onnxruntime/test/2.png', resized_shape=608)
    # outs = l2t.recognize_by_cnstd(r'/home/onnxruntime/test/2.png', resized_shape=608)
    print(outs)
    # print(merge_line_texts(outs, auto_line_break=True))
    