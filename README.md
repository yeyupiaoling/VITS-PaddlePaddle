简体中文 | [English]()

# 基于Pytorch实现的语音合成系统

![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/yeyupiaoling/VITS-Pytorch)
![GitHub Repo stars](https://img.shields.io/github/stars/yeyupiaoling/VITS-Pytorch)
![GitHub](https://img.shields.io/github/license/yeyupiaoling/VITS-Pytorch)
![支持系统](https://img.shields.io/badge/支持系统-Win/Linux/MAC-9cf)

# 前言

本项目是基于Pytorch的语音合成项目，使用的是VITS，这种模型时端到端的，使用起来非常简单。

**欢迎大家扫码入知识星球或者QQ群讨论，知识星球里面提供项目的模型文件和博主其他相关项目的模型文件，也包括其他一些资源。**

<div align="center">
  <img src="https://yeyupiaoling.cn/zsxq.png" alt="知识星球" width="400">
  <img src="https://yeyupiaoling.cn/qq.png" alt="QQ群" width="400">
</div>

# 使用准备

- Anaconda 3
- Python 3.8
- Pytorch 1.13.1
- Windows 10 or Ubuntu 18.04


# 模型下载

|                           数据集                            | 说话人数量 | 说话人名称 |                               下载地址                               | 
|:--------------------------------------------------------:|:-----:|:-----:|:----------------------------------------------------------------:|
| [BZNSYP](https://aistudio.baidu.com/datasetdetail/36741) |   1   | 标准女声  | [点击下载](https://pan.baidu.com/s/1tuHlcfUb0emeszLLD72aNg?pwd=9b5g) |

## 安装环境

- 首先安装的是Pytorch的GPU版本，如果已经安装过了，请跳过。

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

- 安装mvits库。

使用pip安装，命令如下：

```shell
python -m pip install ppvits -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**建议源码安装**，源码安装能保证使用最新代码。

```shell
git clone https://github.com/yeyupiaoling/VITS-Pytorch.git
cd VITS-Pytorch/
python setup.py install
```

## 准备数据

项目支持直接生成[BZNSYP](https://aistudio.baidu.com/datasetdetail/36741)和[AiShell3](https://aistudio.baidu.com/datasetdetail/207703)数据列表，以BZNSYP为例，将BZNSYP下载到`dataset`目录下，并解压。然后执行`create_list.py`程序就会生成以下格式的数据表，格式为`<音频路径>|<说话人名称>|<标注数据>`，注意标注数据需要标注语言，例如简体中文，就要用`[ZH]`将文本包裹起来，其他语言分别支持日本語:`[JA]`, English:[EN], 한국어:[KO]。自定义数据集按照这个格式生成就行。
```
dataset/BZNSYP/Wave/000001.wav|标准女声|[ZH]卡尔普陪外孙玩滑梯。[ZH]
dataset/BZNSYP/Wave/000002.wav|标准女声|[ZH]假语村言别再拥抱我。[ZH]
dataset/BZNSYP/Wave/000003.wav|标准女声|[ZH]宝马配挂跛骡鞍，貂蝉怨枕董翁榻。[ZH]
```

有了数据列表之后，需要生成音素数据列表，只要执行`preprocess_data.py --train_data_list=dataset/bznsyp.txt`，即可生成音素数据列表。到这一步数据就全部准备好了。
```
dataset/BZNSYP/Wave/000001.wav|0|kʰa↓↑əɹ`↓↑pʰu↓↑ pʰeɪ↑ waɪ↓swən→ wan↑ xwa↑tʰi→.
dataset/BZNSYP/Wave/000002.wav|0|tʃ⁼ja↓↑ɥ↓↑ tsʰwən→jɛn↑p⁼iɛ↑ ts⁼aɪ↓ jʊŋ→p⁼ɑʊ↓ wo↓↑.
dataset/BZNSYP/Wave/000003.wav|0|p⁼ɑʊ↓↑ma↓↑ pʰeɪ↓k⁼wa↓ p⁼wo↓↑ lwo↑an→, t⁼iɑʊ→ts`ʰan↑ ɥæn↓ ts`⁼ən↓↑ t⁼ʊŋ↓↑ʊŋ→ tʰa↓.
```


## 训练

现在就可以开始训练模型了，配置文件里面的参数一般不需要修改，说话人数量和说话人名称都会在执行`preprocess_data.py`修改过。可能需要修改的只有`train.batch_size`，如果是显存不够的话，可以减小这个参数。

```shell
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python train.py
# 多卡训练
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
```

训练输出日志：

```
[2023-08-25 16:38:53.893105 INFO   ] utils:print_arguments:101 - ----------- 额外配置参数 -----------
[2023-08-25 16:38:53.893203 INFO   ] utils:print_arguments:103 - config: configs/config.yml
[2023-08-25 16:38:53.893245 INFO   ] utils:print_arguments:103 - epochs: 10000
[2023-08-25 16:38:53.893286 INFO   ] utils:print_arguments:103 - model_dir: models
[2023-08-25 16:38:53.893322 INFO   ] utils:print_arguments:103 - pretrained_model: None
[2023-08-25 16:38:53.893355 INFO   ] utils:print_arguments:103 - resume_model: None
[2023-08-25 16:38:53.893391 INFO   ] utils:print_arguments:104 - ------------------------------------------------
[2023-08-25 16:38:53.932733 INFO   ] utils:print_arguments:106 - ----------- 配置文件参数 -----------
[2023-08-25 16:38:53.932824 INFO   ] utils:print_arguments:109 - data:
[2023-08-25 16:38:53.932866 INFO   ] utils:print_arguments:116 -        add_blank: True
[2023-08-25 16:38:53.932901 INFO   ] utils:print_arguments:116 -        cleaned_text: True
[2023-08-25 16:38:53.932933 INFO   ] utils:print_arguments:116 -        filter_length: 1024
[2023-08-25 16:38:53.932965 INFO   ] utils:print_arguments:116 -        hop_length: 256
[2023-08-25 16:38:53.932997 INFO   ] utils:print_arguments:116 -        max_wav_value: 32768.0
[2023-08-25 16:38:53.933027 INFO   ] utils:print_arguments:116 -        mel_fmax: None
[2023-08-25 16:38:53.933058 INFO   ] utils:print_arguments:116 -        mel_fmin: 0.0
[2023-08-25 16:38:53.933089 INFO   ] utils:print_arguments:116 -        n_mel_channels: 80
[2023-08-25 16:38:53.933120 INFO   ] utils:print_arguments:116 -        n_speakers: 1
[2023-08-25 16:38:53.933150 INFO   ] utils:print_arguments:116 -        num_workers: 4
[2023-08-25 16:38:53.933181 INFO   ] utils:print_arguments:116 -        sampling_rate: 22050
[2023-08-25 16:38:53.933212 INFO   ] utils:print_arguments:116 -        text_cleaners: ['cjke_cleaners2']
[2023-08-25 16:38:53.933243 INFO   ] utils:print_arguments:116 -        training_files: dataset/train.txt
[2023-08-25 16:38:53.933273 INFO   ] utils:print_arguments:116 -        validation_files: dataset/val.txt
[2023-08-25 16:38:53.933303 INFO   ] utils:print_arguments:116 -        win_length: 1024
[2023-08-25 16:38:53.933334 INFO   ] utils:print_arguments:109 - model:
[2023-08-25 16:38:53.933367 INFO   ] utils:print_arguments:116 -        filter_channels: 768
[2023-08-25 16:38:53.933398 INFO   ] utils:print_arguments:116 -        gin_channels: 256
[2023-08-25 16:38:53.933428 INFO   ] utils:print_arguments:116 -        hidden_channels: 192
[2023-08-25 16:38:53.933461 INFO   ] utils:print_arguments:116 -        inter_channels: 192
[2023-08-25 16:38:53.933492 INFO   ] utils:print_arguments:116 -        kernel_size: 3
[2023-08-25 16:38:53.933522 INFO   ] utils:print_arguments:116 -        n_heads: 2
[2023-08-25 16:38:53.933552 INFO   ] utils:print_arguments:116 -        n_layers: 6
[2023-08-25 16:38:53.933582 INFO   ] utils:print_arguments:116 -        n_layers_q: 3
[2023-08-25 16:38:53.933614 INFO   ] utils:print_arguments:116 -        p_dropout: 0.1
[2023-08-25 16:38:53.933644 INFO   ] utils:print_arguments:116 -        resblock: 1
[2023-08-25 16:38:53.933676 INFO   ] utils:print_arguments:116 -        resblock_dilation_sizes: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
[2023-08-25 16:38:53.933706 INFO   ] utils:print_arguments:116 -        resblock_kernel_sizes: [3, 7, 11]
[2023-08-25 16:38:53.933737 INFO   ] utils:print_arguments:116 -        upsample_initial_channel: 512
[2023-08-25 16:38:53.933767 INFO   ] utils:print_arguments:116 -        upsample_kernel_sizes: [16, 16, 4, 4]
[2023-08-25 16:38:53.933798 INFO   ] utils:print_arguments:116 -        upsample_rates: [8, 8, 2, 2]
[2023-08-25 16:38:53.933828 INFO   ] utils:print_arguments:116 -        use_spectral_norm: False
[2023-08-25 16:38:53.933858 INFO   ] utils:print_arguments:109 - speakers:
[2023-08-25 16:38:53.933890 INFO   ] utils:print_arguments:116 -        标准女声: 0
[2023-08-25 16:38:53.933929 INFO   ] utils:print_arguments:118 - symbols: ['_', ',',·································'↑', ' ']
[2023-08-25 16:38:53.933964 INFO   ] utils:print_arguments:109 - train:
[2023-08-25 16:38:53.933997 INFO   ] utils:print_arguments:116 -        batch_size: 16
[2023-08-25 16:38:53.934029 INFO   ] utils:print_arguments:116 -        betas: [0.8, 0.99]
[2023-08-25 16:38:53.934060 INFO   ] utils:print_arguments:116 -        c_kl: 1.0
[2023-08-25 16:38:53.934090 INFO   ] utils:print_arguments:116 -        c_mel: 45
[2023-08-25 16:38:53.934120 INFO   ] utils:print_arguments:116 -        epochs: 10000
[2023-08-25 16:38:53.934151 INFO   ] utils:print_arguments:116 -        eps: 1e-09
[2023-08-25 16:38:53.934181 INFO   ] utils:print_arguments:116 -        eval_interval: 1000
[2023-08-25 16:38:53.934211 INFO   ] utils:print_arguments:116 -        fp16_run: True
[2023-08-25 16:38:53.934244 INFO   ] utils:print_arguments:116 -        init_lr_ratio: 1
[2023-08-25 16:38:53.934276 INFO   ] utils:print_arguments:116 -        learning_rate: 0.0002
[2023-08-25 16:38:53.934305 INFO   ] utils:print_arguments:116 -        log_interval: 200
[2023-08-25 16:38:53.934336 INFO   ] utils:print_arguments:116 -        lr_decay: 0.999875
[2023-08-25 16:38:53.934366 INFO   ] utils:print_arguments:116 -        seed: 1234
[2023-08-25 16:38:53.934396 INFO   ] utils:print_arguments:116 -        segment_size: 8192
[2023-08-25 16:38:53.934426 INFO   ] utils:print_arguments:116 -        warmup_epochs: 0
[2023-08-25 16:38:53.934455 INFO   ] utils:print_arguments:119 - ------------------------------------------------
[2023-08-25 16:38:53.985538 INFO   ] trainer:__setup_dataloader:69 - 训练数据：9901
epoch [1/10000]: 100%|██████████| 619/619 [05:30<00:00,  1.88it/s]]
[2023-08-25 16:44:25.205557 INFO   ] trainer:train:168 - ======================================================================
epoch [2/10000]: 100%|██████████| 619/619 [05:20<00:00,  1.93it/s]s]
[2023-08-25 16:49:54.372718 INFO   ] trainer:train:168 - ======================================================================
epoch [3/10000]: 100%|██████████| 619/619 [05:19<00:00,  1.94it/s]
[2023-08-25 16:55:21.277194 INFO   ] trainer:train:168 - ======================================================================
epoch [4/10000]: 100%|██████████| 619/619 [05:18<00:00,  1.94it/s]
```

训练的日志也会使用VisualDL保存，可以使用这个工具实时查看loss变化和合成效果，只要在项目根目录执行`visualdl --logdir=log/ --host=0.0.0.0`，访问`http://<IP地址>:8040`即可打开页面，效果如下。

<div align="center">
    <img src="./docs/images/log.jpg" alt="VisualDL" width="600">
</div>

# 语音合成

训练到一定程度之后，可以开始使用模型进行语音合成了，命令如下，主要参数有三个，分别是`--language`指定合成文本的语言，然后是`--text`指定所需要合成的文本，最后是指定说话人的参数`--spk`。快去试一下吧。

```shell
python infer.py --language=简体中文 --text="你好，我是智能语音助手。" --spk=标准女声
```

## 打赏作者

<br/>
<div align="center">
    <p>打赏一块钱支持一下作者</p>
    <img src="https://yeyupiaoling.cn/reward.png" alt="打赏作者" width="400">
</div>

# 参考资料

1. https://github.com/Plachtaa/VITS-fast-fine-tuning
2. https://github.com/PaddlePaddle/PaddleSpeech
3. https://github.com/yeyupiaoling/MASR
