import re
from typing import Tuple, Optional, Dict, Any
import logging
import yaml
from pathlib import Path
import os
from munch import Munch
import cv2

from inspect import isfunction
import contextlib

from PIL import Image
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

from latex.hybrid import get_encoder as hybrid_get_encoder
from latex.vit import get_encoder as vit_get_encoder
from latex.lt_transformer import get_decoder as lt_transformer_get_encoder

LATEX_CONFIG_FP = Path(__file__).parent.absolute() / 'latex_config.yaml'


logger = logging.getLogger(__name__)

import platform
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


# def download_checkpoints(args):
#     # adapted from pix2tex.model.checkpoints.get_latest_checkpoint
#     ckpt_list = [args.mfr_checkpoint, args.resizer_checkpoint]
#     tag = 'v0.0.1'  # get_latest_tag()
#     weights = (
#         'https://github.com/lukas-blecher/LaTeX-OCR/releases/download/%s/weights.pth'
#         % tag
#     )
#     resizer = (
#         'https://github.com/lukas-blecher/LaTeX-OCR/releases/download/%s/image_resizer.pth'
#         % tag
#     )
#     for idx, (url, fp) in enumerate(zip([weights, resizer], ckpt_list)):
#         name = os.path.basename(url)
#         if not os.path.exists(fp):
#             if os.path.basename(fp) != name:
#                 logger.warning(f'can not find file {fp}, download {name} from {url} instead')
#                 fp = os.path.join(os.path.dirname(fp), name)
#                 if idx == 0:
#                     args.mfr_checkpoint = fp
#                 else:
#                     args.resizer_checkpoint = fp
#             os.makedirs(os.path.dirname(fp), exist_ok=True)
#             file = download_as_bytes_with_progress(url, name)
#             logger.info('downloading file %s to path %s', name, fp)
#             open(fp, "wb").write(file)
#             logger.info(f'save {name} to path {fp}')


def minmax_size(
    img: Image,
    max_dimensions: Tuple[int, int] = None,
    min_dimensions: Tuple[int, int] = None,
) -> Image:
    """Resize or pad an image to fit into given dimensions

    Args:
        img (Image): Image to scale up/down.
        max_dimensions (Tuple[int, int], optional): Maximum dimensions. Defaults to None.
        min_dimensions (Tuple[int, int], optional): Minimum dimensions. Defaults to None.

    Returns:
        Image: Image with correct dimensionality
    """
    if max_dimensions is not None:
        ratios = [a / b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size) // max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
        padded_size = [
            max(img_dim, min_dim) for img_dim, min_dim in zip(img.size, min_dimensions)
        ]
        if padded_size != list(img.size):  # assert hypothesis
            padded_im = Image.new('L', padded_size, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


def find_all_left_or_right(latex, left_or_right='left'):
    left_bracket_infos = []
    prefix_len = len(left_or_right) + 1
    # 匹配出latex中所有的 '\left' 后面跟着的第一个非空格字符，定位它们所在的位置
    for m in re.finditer(rf'\\{left_or_right}\s*\S', latex):
        start, end = m.span()
        # 如果最后一个字符为 "\"，则往前继续匹配，直到匹配到一个非字母的字符
        # 如 "\left \big("
        while latex[end - 1] in ('\\', ' '):
            end += 1
            while end < len(latex) and latex[end].isalpha():
                end += 1
        ori_str = latex[start + prefix_len : end].strip()
        # FIXME: ori_str中可能出现多个 '\left'，此时需要分隔开

        left_bracket_infos.append({'str': ori_str, 'start': start, 'end': end})
        left_bracket_infos.sort(key=lambda x: x['start'])
    return left_bracket_infos


def match_left_right(left_str, right_str):
    """匹配左右括号，如匹配 `\left(` 和 `\right)`。"""
    left_str = left_str.strip().replace(' ', '')[len('left') + 1 :]
    right_str = right_str.strip().replace(' ', '')[len('right') + 1 :]
    # 去掉开头的相同部分
    while left_str and right_str and left_str[0] == right_str[0]:
        left_str = left_str[1:]
        right_str = right_str[1:]

    match_pairs = [
        ('', ''),
        ('(', ')'),
        ('\{', '.'),  # 大括号那种
        ('⟮', '⟯'),
        ('[', ']'),
        ('⟨', '⟩'),
        ('{', '}'),
        ('⌈', '⌉'),
        ('┌', '┐'),
        ('⌊', '⌋'),
        ('└', '┘'),
        ('⎰', '⎱'),
        ('lt', 'gt'),
        ('lang', 'rang'),
        (r'langle', r'rangle'),
        (r'lbrace', r'rbrace'),
        ('lBrace', 'rBrace'),
        (r'lbracket', r'rbracket'),
        (r'lceil', r'rceil'),
        ('lcorner', 'rcorner'),
        (r'lfloor', r'rfloor'),
        (r'lgroup', r'rgroup'),
        (r'lmoustache', r'rmoustache'),
        (r'lparen', r'rparen'),
        (r'lvert', r'rvert'),
        (r'lVert', r'rVert'),
    ]
    return (left_str, right_str) in match_pairs


def post_post_process_latex(latex: str) -> str:
    """对识别结果做进一步处理和修正。"""
    # 把latex中的中文括号全部替换成英文括号
    latex = latex.replace('（', '(').replace('）', ')')
    # 把latex中的中文逗号全部替换成英文逗号
    latex = latex.replace('，', ',')

    left_bracket_infos = find_all_left_or_right(latex, left_or_right='left')
    right_bracket_infos = find_all_left_or_right(latex, left_or_right='right')
    # left 和 right 找配对，left找位置比它靠前且最靠近他的right配对
    for left_bracket_info in left_bracket_infos:
        for right_bracket_info in right_bracket_infos:
            if (
                not right_bracket_info.get('matched', False)
                and right_bracket_info['start'] > left_bracket_info['start']
                and match_left_right(
                    right_bracket_info['str'], left_bracket_info['str']
                )
            ):
                left_bracket_info['matched'] = True
                right_bracket_info['matched'] = True
                break

    for left_bracket_info in left_bracket_infos:
        # 把没有匹配的 '\left'替换为等长度的空格
        left_len = len('left') + 1
        if not left_bracket_info.get('matched', False):
            start_idx = left_bracket_info['start']
            end_idx = start_idx + left_len
            latex = (
                latex[: left_bracket_info['start']]
                + ' ' * (end_idx - start_idx)
                + latex[end_idx:]
            )
    for right_bracket_info in right_bracket_infos:
        # 把没有匹配的 '\right'替换为等长度的空格
        right_len = len('right') + 1
        if not right_bracket_info.get('matched', False):
            start_idx = right_bracket_info['start']
            end_idx = start_idx + right_len
            latex = (
                latex[: right_bracket_info['start']]
                + ' ' * (end_idx - start_idx)
                + latex[end_idx:]
            )

    # 把 latex 中的连续空格替换为一个空格
    latex = re.sub(r'\s+', ' ', latex)
    return latex

def get_device(args, no_cuda=False):
    device = 'cpu'
    available_gpus = torch.cuda.device_count()
    args.gpu_devices = args.gpu_devices if args.get('gpu_devices', False) else list(range(available_gpus))
    if available_gpus > 0 and not no_cuda:
        device = 'cuda:%d' % args.gpu_devices[0] if args.gpu_devices else 0
        assert available_gpus >= len(args.gpu_devices), "Available %d gpu, but specified gpu %s." % (available_gpus, ','.join(map(str, args.gpu_devices)))
        assert max(args.gpu_devices) < available_gpus, "legal gpu_devices should in [%s], received [%s]" % (','.join(map(str, range(available_gpus))), ','.join(map(str, args.gpu_devices)))
    return device
def parse_args(args, **kwargs) -> Munch:
    args = Munch({'epoch': 0}, **args)
    kwargs = Munch({'no_cuda': False, 'debug': False}, **kwargs)
    args.update(kwargs)
    args.wandb = not kwargs.debug and not args.debug
    args.device = get_device(args, kwargs.no_cuda)
    args.encoder_structure = args.get('encoder_structure', 'hybrid')
    args.max_dimensions = [args.max_width, args.max_height]
    args.min_dimensions = [args.get('min_width', 32), args.get('min_height', 32)]
    if 'decoder_args' not in args or args.decoder_args is None:
        args.decoder_args = {}
    return args

import torch
import torch.nn as nn

import numpy as np

class Model(nn.Module):
    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def data_parallel(self, x: torch.Tensor, device_ids, output_device=None, **kwargs):
        if not device_ids or len(device_ids) == 1:
            return self(x, **kwargs)
        if output_device is None:
            output_device = device_ids[0]
        replicas = nn.parallel.replicate(self, device_ids)
        inputs = nn.parallel.scatter(x, device_ids)  # Slices tensors into approximately equal chunks and distributes them across given GPUs.
        kwargs = nn.parallel.scatter(kwargs, device_ids)  # Duplicates references to objects that are not tensors.
        replicas = replicas[:len(inputs)]
        kwargs = kwargs[:len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs, kwargs)
        return nn.parallel.gather(outputs, output_device).mean()

    def forward(self, x: torch.Tensor, tgt_seq: torch.Tensor,  **kwargs):
        encoded = self.encoder(x)
        out = self.decoder(tgt_seq, context=encoded, **kwargs)
        return out

    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float = 0.25):
        return self.decoder.generate((torch.LongTensor([self.args.bos_token]*len(x))[:, None]).to(x.device), self.args.max_seq_len,
                                     eos_token=self.args.eos_token, context=self.encoder(x), temperature=temperature)


def get_model(args):
    if args.encoder_structure.lower() == 'vit':
        encoder = vit_get_encoder(args)
    elif args.encoder_structure.lower() == 'hybrid':
        encoder = hybrid_get_encoder(args)
    else:
        raise NotImplementedError('Encoder structure "%s" not supported.' % args.encoder_structure)
    decoder = lt_transformer_get_encoder(args)
    encoder.to(args.device)
    decoder.to(args.device)
    model = Model(encoder, decoder, args)
    # if args.wandb:
    #     import wandb
    #     wandb.watch(model)

    return model

import albumentations as alb
from albumentations.pytorch import ToTensorV2

train_transform = alb.Compose(
    [
        alb.Compose(
            [alb.ShiftScaleRotate(shift_limit=0, scale_limit=(-.15, 0), rotate_limit=1, border_mode=0, interpolation=3,
                                  value=[255, 255, 255], p=1),
             alb.GridDistortion(distort_limit=0.1, border_mode=0, interpolation=3, value=[255, 255, 255], p=.5)], p=.15),
        # alb.InvertImg(p=.15),
        alb.RGBShift(r_shift_limit=15, g_shift_limit=15,
                     b_shift_limit=15, p=0.3),
        alb.GaussNoise(10, p=.2),
        alb.RandomBrightnessContrast(.05, (-.2, 0), True, p=0.2),
        alb.ImageCompression(95, p=.3),
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)
test_transform = alb.Compose(
    [
        alb.ToGray(always_apply=True),
        alb.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738)),
        # alb.Sharpen()
        ToTensorV2(),
    ]
)

def pad(img: Image, divable: int = 32) -> Image:
    """Pad an Image to the next full divisible value of `divable`. Also normalizes the image and invert if needed.

    Args:
        img (PIL.Image): input image
        divable (int, optional): . Defaults to 32.

    Returns:
        PIL.Image
    """
    threshold = 128
    data = np.array(img.convert('LA'))
    if data[..., -1].var() == 0:
        data = (data[..., 0]).astype(np.uint8)
    else:
        data = (255-data[..., -1]).astype(np.uint8)
    data = (data-data.min())/(data.max()-data.min())*255
    if data.mean() > threshold:
        # To invert the text to white
        gray = 255*(data < threshold).astype(np.uint8)
    else:
        gray = 255*(data > threshold).astype(np.uint8)
        data = 255-data

    coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
    a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = data[b:b+h, a:a+w]
    im = Image.fromarray(rect).convert('L')
    dims = []
    for x in [w, h]:
        div, mod = divmod(x, divable)
        dims.append(divable*(div + (1 if mod > 0 else 0)))
    padded = Image.new('L', dims, 255)
    padded.paste(im, (0, 0, im.size[0], im.size[1]))
    return padded

def post_process(s: str):
    """Remove unnecessary whitespace from LaTeX code.

    Args:
        s (str): Input string

    Returns:
        str: Processed image
    """
    text_reg = r'(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})'
    letter = '[a-zA-Z]'
    noletter = '[\W_^\d]'
    names = [x[0].replace(' ', '') for x in re.findall(text_reg, s)]
    s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
    news = s
    while True:
        s = news
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, noletter), r'\1\2', s)
        news = re.sub(r'(?!\\ )(%s)\s+?(%s)' % (noletter, letter), r'\1\2', news)
        news = re.sub(r'(%s)\s+?(%s)' % (letter, noletter), r'\1\2', news)
        if news == s:
            break
    return s


def token2str(tokens, tokenizer) -> list:
    if len(tokens.shape) == 1:
        tokens = tokens[None, :]
    dec = [tokenizer.decode(tok) for tok in tokens]
    return [''.join(detok.split(' ')).replace('Ġ', ' ').replace('[EOS]', '').replace('[BOS]', '').replace('[PAD]', '').strip() for detok in dec]

class LatexOCR(object):
    """Get a prediction of an image in the easiest way"""

    image_resizer = None
    last_pic = None

    def __init__(self, arguments: Optional[Dict[str, Any]] = None):
        """Initialize a LatexOCR model

        Args:
            arguments (Union[Namespace, Munch], optional): Special model parameters. Defaults to None.
        """
        def_arguments = {
            'config': LATEX_CONFIG_FP,
            'mfr_checkpoint': Path(data_dir()) / 'formula' / 'weights.pth',
            'resizer_checkpoint': Path(data_dir()) / 'formula' / 'image_resizer.pth',
            # 'no_cuda': True,
            'no_resize': False,
            'device': 'cpu',
        }
        if arguments is not None:
            if 'model_fp' in arguments:
                arguments['mfr_checkpoint'] = arguments.pop('model_fp')
            def_arguments.update(arguments)
        arguments = def_arguments

        arguments = Munch(arguments)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        with open(arguments.config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.args = parse_args(Munch(params))
        self.args.update(**vars(arguments))
        self.args.wandb = False
        # self.args.device = (
        #     'cuda' if torch.cuda.is_available() and not self.args.no_cuda else 'cpu'
        # )
        # download_checkpoints(self.args)

        self.model = get_model(self.args)
        self.model.load_state_dict(
            torch.load(self.args.mfr_checkpoint, map_location=self.args.device)
        )
        logger.info(f'use model: {self.args.mfr_checkpoint}')
        self.model.eval()

        if not self.args.no_resize and os.path.isfile(self.args.resizer_checkpoint):
            self.image_resizer = ResNetV2(
                layers=[2, 3, 3],
                num_classes=max(self.args.max_dimensions) // 32,
                global_pool='avg',
                in_chans=1,
                drop_rate=0.05,
                preact=True,
                stem_type='same',
                conv_layer=StdConv2dSame,
            ).to(self.args.device)
            self.image_resizer.load_state_dict(
                torch.load(
                    self.args.resizer_checkpoint,
                    map_location=self.args.device,
                )
            )
            logger.info(f'use model: {self.args.resizer_checkpoint}')
            self.image_resizer.eval()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.args.tokenizer)

    def __call__(self, img=None, resize=True) -> str:
        """Get a prediction from an image

        Args:
            img (Image, optional): Image to predict. Defaults to None.
            resize (bool, optional): Whether to call the resize model. Defaults to True.

        Returns:
            str: predicted Latex code
        """
        if type(img) is bool:
            img = None
        if img is None:
            if self.last_pic is None:
                print('Provide an image.')
                return ''
            else:
                img = self.last_pic.copy()
        else:
            self.last_pic = img.copy()
        img = minmax_size(pad(img), self.args.max_dimensions, self.args.min_dimensions)
        if (self.image_resizer is not None and not self.args.no_resize) and resize:
            with torch.no_grad():
                input_image = img.convert('RGB').copy()
                r, w, h = 1, input_image.size[0], input_image.size[1]
                for _ in range(10):
                    h = int(h * r)  # height to resize
                    try:
                        img = pad(
                            minmax_size(
                                input_image.resize(
                                    (w, h),
                                    Image.Resampling.BILINEAR
                                    if r > 1
                                    else Image.Resampling.LANCZOS,
                                ),
                                self.args.max_dimensions,
                                self.args.min_dimensions,
                            )
                        )
                        t = test_transform(image=np.array(img.convert('RGB')))['image'][
                            :1
                        ].unsqueeze(0)
                        w = (
                            self.image_resizer(t.to(self.args.device)).argmax(-1).item()
                            + 1
                        ) * 32
                        logger.debug((r, img.size, (w, int(input_image.size[1] * r))))
                    except Exception as e:
                        logger.warning(e)
                        break

                    if w == img.size[0]:
                        break
                    r = w / img.size[0]
        else:
            img = np.array(pad(img).convert('RGB'))
            t = test_transform(image=img)['image'][:1].unsqueeze(0)
        im = t.to(self.args.device)

        dec = self.model.generate(
            im.to(self.args.device), temperature=self.args.get('temperature', 0.25)
        )
        pred = post_process(token2str(dec, self.tokenizer)[0])
        pred = post_post_process_latex(pred)
        return pred