import os

import paddle
import yaml
from paddle import no_grad

from ppvits.models.commons import intersperse
from ppvits.models.models import SynthesizerTrn
from ppvits.text import text_to_sequence, get_symbols
from ppvits.utils.logger import setup_logger
from ppvits.utils.utils import load_checkpoint, print_arguments, dict_to_object

logger = setup_logger(__name__)


class PPVITSPredictor:
    def __init__(self, configs, model_path, use_gpu=True):
        if use_gpu:
            assert paddle.is_compiled_with_cuda(), 'GPU不可用'
            paddle.device.set_device("gpu")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            paddle.device.set_device("cpu")
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        self.symbols = get_symbols(self.configs.dataset_conf.text_cleaner)
        checkpoint_dict = paddle.load(model_path)
        self.speaker_ids = checkpoint_dict['speakers']
        # 获取模型
        self.net_g = SynthesizerTrn(len(self.symbols),
                                    self.configs.dataset_conf.filter_length // 2 + 1,
                                    self.configs.train.segment_size // self.configs.dataset_conf.hop_length,
                                    n_speakers=len(self.speaker_ids),
                                    **self.configs.model)
        self.net_g.eval()
        load_checkpoint(model_path, self.net_g, None)
        # 获取支持语言
        if self.configs.dataset_conf.text_cleaner in LANGUAGE_MARKS.keys():
            self.language_marks = LANGUAGE_MARKS[self.configs.dataset_conf.text_cleaner]
        else:
            raise Exception(f"不支持方法：{self.configs.dataset_conf.text_cleaner}")
        logger.info(f'支持说话人：{list(self.speaker_ids.keys())}')

    def get_text(self, text, config, is_symbol):
        text_norm = text_to_sequence(text, self.symbols, [] if is_symbol else config.dataset_conf.text_cleaner)
        if config.dataset_conf.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = paddle.to_tensor(data=text_norm, dtype=paddle.int64)
        return text_norm

    def generate(self, text, spk, language, noise_scale=0.667, noise_scale_w=0.6, speed=1):
        assert spk in self.speaker_ids.keys(), f'不存在说话人：{spk}'
        assert language in self.language_marks.keys(), f'不支持语言：{language}'
        # 输入到模型的文本
        text = self.language_marks[language] + text + self.language_marks[language]
        speaker_id = self.speaker_ids[spk]
        stn_tst = self.get_text(text, self.configs, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = paddle.to_tensor(data=[stn_tst.shape[0]], dtype=paddle.int64)
            sid = paddle.to_tensor(data=[speaker_id], dtype=paddle.int64)
            audio = self.net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                     noise_scale_w=noise_scale_w,
                                     length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return audio, self.configs.dataset_conf.sampling_rate
