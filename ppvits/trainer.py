import os
import platform
import shutil

import paddle
import yaml
from paddle.distributed import fleet
from paddle.io import DataLoader
from paddle.nn import functional as F
from tqdm import tqdm
from visualdl import LogWriter

from ppvits.data_utils.collate_fn import TextAudioSpeakerCollate
from ppvits.data_utils.mel_processing import mel_spectrogram_paddle, spec_to_mel_paddle
from ppvits.data_utils.reader import TextAudioSpeakerLoader
from ppvits.data_utils.sampler import DistributedBucketSampler
from ppvits.models.commons import clip_grad_value_, slice_segments
from ppvits.models.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from ppvits.models.models import SynthesizerTrn, MultiPeriodDiscriminator
from ppvits.utils.logger import setup_logger
from ppvits.utils.utils import load_checkpoint, save_checkpoint, plot_spectrogram_to_numpy, dict_to_object, \
    print_arguments

logger = setup_logger(__name__)


class PPVITSTrainer(object):
    def __init__(self, configs, model_dir):
        assert paddle.is_compiled_with_cuda(), "CPU training is not allowed."
        self.model_dir = model_dir
        # 读取配置文件
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        self.symbols = self.configs['symbols']
        # 复制一份
        os.makedirs(model_dir, exist_ok=True)
        config_save_path = os.path.join(model_dir, "config.yml")
        with open(config_save_path, "w", encoding='utf-8') as f:
            yaml_datas = yaml.dump(configs, indent=2, sort_keys=False, allow_unicode=True)
            f.write(yaml_datas)
        if platform.system().lower() == 'windows':
            self.configs.data.num_workers = 0
            logger.warning('Windows系统不支持多线程读取数据，已自动关闭！')

    def __setup_dataloader(self, rank, n_gpus):
        train_dataset = TextAudioSpeakerLoader(self.configs.data.training_files, self.configs.data, self.symbols)
        train_sampler = None
        if n_gpus > 1:
            train_sampler = DistributedBucketSampler(train_dataset,
                                                     self.configs.train.batch_size,
                                                     [32, 300, 400, 500, 600, 700, 800, 900, 1000],
                                                     num_replicas=n_gpus,
                                                     rank=rank,
                                                     shuffle=True)
        collate_fn = TextAudioSpeakerCollate()
        self.train_loader = DataLoader(train_dataset, num_workers=self.configs.data.num_workers,
                                       shuffle=(train_sampler is None), collate_fn=collate_fn,
                                       batch_sampler=train_sampler, batch_size=self.configs.train.batch_size)
        if rank == 0:
            eval_dataset = TextAudioSpeakerLoader(self.configs.data.validation_files, self.configs.data, self.symbols)
            self.eval_loader = DataLoader(eval_dataset, num_workers=self.configs.data.num_workers, shuffle=False,
                                          batch_size=self.configs.train.batch_size, drop_last=False,
                                          collate_fn=collate_fn)
        logger.info('训练数据：{}'.format(len(train_dataset)))

    def __setup_model(self, n_gpus, resume_model=None, pretrained_model=None):
        latest_epoch = 0
        self.net_g = SynthesizerTrn(len(self.symbols),
                                    self.configs.data.filter_length // 2 + 1,
                                    self.configs.train.segment_size // self.configs.data.hop_length,
                                    n_speakers=self.configs.data.n_speakers,
                                    **self.configs.model)
        self.net_d = MultiPeriodDiscriminator(self.configs.model.use_spectral_norm)
        # 学习率衰减
        self.scheduler_g = paddle.optimizer.lr.ExponentialDecay(learning_rate=self.configs.train.learning_rate,
                                                                gamma=self.configs.train.lr_decay)
        self.scheduler_d = paddle.optimizer.lr.ExponentialDecay(learning_rate=self.configs.train.learning_rate,
                                                                gamma=self.configs.train.lr_decay)
        # 获取优化方法
        self.optim_g = paddle.optimizer.AdamW(parameters=self.net_g.parameters(),
                                              learning_rate=self.scheduler_g,
                                              beta1=self.configs.train.betas[0],
                                              beta2=self.configs.train.betas[1],
                                              epsilon=self.configs.train.eps)
        self.optim_d = paddle.optimizer.AdamW(parameters=self.net_d.parameters(),
                                              learning_rate=self.scheduler_d,
                                              beta1=self.configs.train.betas[0],
                                              beta2=self.configs.train.betas[1],
                                              epsilon=self.configs.train.eps)
        # 自动恢复训练
        latest_g_checkpoint_path = os.path.join(self.model_dir, "latest", "g_net.pdparams")
        latest_d_checkpoint_path = os.path.join(self.model_dir, "latest", "d_net.pdparams")
        if os.path.exists(latest_g_checkpoint_path):
            _, _, _, latest_epoch = load_checkpoint(latest_g_checkpoint_path, self.net_g, self.optim_g)
        if os.path.exists(latest_d_checkpoint_path):
            _, _, _, latest_epoch = load_checkpoint(latest_d_checkpoint_path, self.net_d, self.optim_d)
        # 加载预训练模型
        if pretrained_model:
            pretrained_g_model_path = os.path.join(pretrained_model, "g_net.pdparams")
            pretrained_d_model_path = os.path.join(pretrained_model, "d_net.pdparams")
            if os.path.exists(pretrained_g_model_path):
                load_checkpoint(pretrained_g_model_path, self.net_g, None)
            if os.path.exists(pretrained_d_model_path):
                load_checkpoint(pretrained_d_model_path, self.net_d, None)
        # 加载恢复训练模型
        if resume_model:
            resume_g_model_path = os.path.join(resume_model, "g_net.pdparams")
            resume_d_model_path = os.path.join(resume_model, "d_net.pdparams")
            if os.path.exists(resume_g_model_path):
                _, _, _, latest_epoch = load_checkpoint(resume_g_model_path, self.net_g, self.optim_g)
            if os.path.exists(resume_d_model_path):
                _, _, _, latest_epoch = load_checkpoint(resume_d_model_path, self.net_d, self.optim_d)
        # freeze all other layers except speaker embedding
        for p in self.net_g.parameters():
            p.requires_grad = True
        for p in self.net_d.parameters():
            p.requires_grad = True
        # for p in net_d.parameters():
        #     p.requires_grad = False
        # net_g.emb_g.weight.requires_grad = True
        # 多卡训练
        if n_gpus > 1:
            self.optim_g = fleet.distributed_optimizer(self.optim_g)
            self.optim_d = fleet.distributed_optimizer(self.optim_d)
            self.net_g = fleet.distributed_model(self.net_g)
            self.net_d = fleet.distributed_model(self.net_d)

        # 半精度训练
        self.amp_scaler = paddle.amp.GradScaler(enable=self.configs.train.fp16_run)
        return latest_epoch

    def train(self, epochs, resume_model=None, pretrained_model=None):
        # 获取有多少张显卡训练
        n_gpus = paddle.distributed.get_world_size()
        writer = None
        rank = paddle.distributed.get_rank()
        if n_gpus > 1:
            # 初始化Fleet环境
            strategy = fleet.DistributedStrategy()
            fleet.init(is_collective=True, strategy=strategy)
        if rank == 0:
            # 日志记录器
            writer = LogWriter(logdir='log')

        # 获取数据
        self.__setup_dataloader(rank=rank, n_gpus=n_gpus)
        # 获取模型
        latest_epoch = self.__setup_model(n_gpus=n_gpus,
                                          resume_model=resume_model,
                                          pretrained_model=pretrained_model)
        self.train_step = 0
        if rank == 0:
            writer.add_scalar('Train/lr_g', self.scheduler_g.get_lr(), latest_epoch)
            writer.add_scalar('Train/lr_d', self.scheduler_d.get_lr(), latest_epoch)
        # 开始训练
        for epoch_id in range(latest_epoch, epochs):
            epoch_id += 1
            # 训练一个epoch
            self.__train_epoch(rank=rank, writer=writer, epoch=epoch_id, max_epochs=epochs)
            self.scheduler_g.step()
            self.scheduler_d.step()
            # 多卡训练只使用一个进程执行评估和保存模型
            if rank == 0:
                writer.add_scalar('Train/lr_g', self.scheduler_g.get_lr(), epoch_id)
                writer.add_scalar('Train/lr_d', self.scheduler_d.get_lr(), epoch_id)
                logger.info('=' * 70)
                self.evaluate(generator=self.net_g, eval_loader=self.eval_loader, writer=writer, epoch=epoch_id)
                # 保存模型
                self.__save_model(epoch_id=epoch_id)

    # 保存模型
    def __save_model(self, epoch_id):
        save_dir = os.path.join(self.model_dir, f"epoch_{epoch_id}")
        latest_dir = os.path.join(self.model_dir, "latest")
        # 保存模型
        save_checkpoint(self.net_g, self.optim_g, self.configs.train.learning_rate, epoch_id,
                        os.path.join(save_dir, "g_net.pdparams"))
        save_checkpoint(self.net_d, self.optim_d, self.configs.train.learning_rate, epoch_id,
                        os.path.join(save_dir, "d_net.pdparams"))
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        shutil.copytree(save_dir, latest_dir)
        # 删除旧模型
        old_model_path = os.path.join(self.model_dir, f"epoch_{epoch_id - 3}")
        if os.path.exists(old_model_path):
            shutil.rmtree(old_model_path)

    def __train_epoch(self, rank, writer, epoch, max_epochs):
        self.net_g.train()
        self.net_d.train()
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) \
                in enumerate(tqdm(self.train_loader, desc=f'epoch [{epoch}/{max_epochs}]')):

            with paddle.amp.auto_cast(enable=self.configs.train.fp16_run):
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                    (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(x, x_lengths, spec, spec_lengths, speakers)

                mel = spec_to_mel_paddle(spec,
                                         self.configs.data.filter_length,
                                         self.configs.data.n_mel_channels,
                                         self.configs.data.sampling_rate,
                                         self.configs.data.mel_fmin,
                                         self.configs.data.mel_fmax)
                y_mel = slice_segments(mel, ids_slice, self.configs.train.segment_size // self.configs.data.hop_length)
                y_hat_mel = mel_spectrogram_paddle(y_hat.squeeze(1),
                                                   self.configs.data.filter_length,
                                                   self.configs.data.n_mel_channels,
                                                   self.configs.data.sampling_rate,
                                                   self.configs.data.hop_length,
                                                   self.configs.data.win_length,
                                                   self.configs.data.mel_fmin,
                                                   self.configs.data.mel_fmax)

                y = slice_segments(y, ids_slice * self.configs.data.hop_length, self.configs.train.segment_size)

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
                with paddle.amp.auto_cast(enable=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    loss_disc_all = loss_disc
            self.optim_d.clear_grad()
            self.amp_scaler.scale(loss_disc_all).backward()
            self.amp_scaler.unscale_(self.optim_d)
            clip_grad_value_(self.net_d.parameters(), None)
            self.amp_scaler.step(self.optim_d)

            with paddle.amp.auto_cast(enable=self.configs.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
                with paddle.amp.auto_cast(enable=False):
                    loss_dur = paddle.sum(l_length.float())
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.configs.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.configs.train.c_kl

                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
            self.optim_g.clear_grad()
            self.amp_scaler.scale(loss_gen_all).backward()
            self.amp_scaler.unscale_(self.optim_g)
            grad_norm_g = clip_grad_value_(self.net_g.parameters(), None)
            self.amp_scaler.step(self.optim_g)
            self.amp_scaler.update()

            if rank == 0 and batch_idx % self.configs.train.log_interval == 0:
                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur,
                                    "loss/g/kl": loss_kl})
                scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                # 可视化
                for k, v in scalar_dict.items():
                    writer.add_scalar(k, v, self.train_step)
                self.train_step += 1

    def evaluate(self, generator, eval_loader, writer, epoch):
        generator.eval()
        if isinstance(generator, paddle.DataParallel):
            eval_generator = generator._layers
        else:
            eval_generator = generator
        with paddle.no_grad():
            for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
                # remove else
                x = x[:1]
                x_lengths = x_lengths[:1]
                spec = spec[:1]
                y = y[:1]
                y_lengths = y_lengths[:1]
                speakers = speakers[:1]
                break
            y_hat, attn, mask, *_ = eval_generator.infer(x, x_lengths, speakers, max_len=1000)
            y_hat_lengths = mask.sum([1, 2]).long() * self.configs.data.hop_length

            mel = spec_to_mel_paddle(spec,
                                     self.configs.data.filter_length,
                                     self.configs.data.n_mel_channels,
                                     self.configs.data.sampling_rate,
                                     self.configs.data.mel_fmin,
                                     self.configs.data.mel_fmax)
            y_hat_mel = mel_spectrogram_paddle(y_hat.squeeze(1).float(),
                                               self.configs.data.filter_length,
                                               self.configs.data.n_mel_channels,
                                               self.configs.data.sampling_rate,
                                               self.configs.data.hop_length,
                                               self.configs.data.win_length,
                                               self.configs.data.mel_fmin,
                                               self.configs.data.mel_fmax)
        image_dict = {"gen/mel": plot_spectrogram_to_numpy(y_hat_mel[0].numpy())}
        audio_dict = {"gen/audio": y_hat[0, :, :y_hat_lengths[0]]}
        image_dict.update({"gt/mel": plot_spectrogram_to_numpy(mel[0].numpy())})
        audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})
        # 可视化
        for k, v in image_dict.items():
            writer.add_image(k, v, epoch, dataformats='HWC')
        for k, v in audio_dict.items():
            writer.add_audio(k, v, epoch, self.configs.data.sampling_rate)

        generator.train()
