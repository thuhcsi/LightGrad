# LightGrad: Lightweight Diffusion Probabilistic Model for Text-to-speech
Demos are available at: https://thuhcsi.github.io/LightGrad/

## Setup Environment

Install python 3.10.

Then, run:
```bash
git clone --recursive https://github.com/thuhcsi/LightGrad.git
python -m pip install -r requirements.txt
```

## Training
### Preprocess for BZNSYP

Download dataset from [url](https://www.data-baker.com/data/index/TNtts).
Run
```bash
python preprocess.py bznsyp [PATH_TO_DIRECTORY_CONTAINING_DATASET] \
    [PATH_TO_DIRECTORY_FOR_SAVING_PREPROCESS_RESULTS] \
    --test_sample_count 200 --valid_sample_count 200
```
This will produce `phn2id.json`, `train_dataset.json`, `test_dataset.json`, `valid_dataset.json` in `[PATH_TO_DIRECTORY_FOR_SAVING_PREPROCESS_RESULTS]`.

### Preprocess for LJSpeech

Download dataset from [url](https://keithito.com/LJ-Speech-Dataset/).
Run
```bash
python preprocess.py ljspeech [PATH_TO_DIRECTORY_CONTAINING_DATASET] \
    [PATH_TO_DIRECTORY_FOR_SAVING_PREPROCESS_RESULTS] \
    --test_sample_count 200 --valid_sample_count 200
```
This will produce `phn2id.json`, `train_dataset.json`, `test_dataset.json`, `valid_dataset.json` in `[PATH_TO_DIRECTORY_FOR_SAVING_PREPROCESS_RESULTS]`.

### Training for BZNSYP

Edit `config/bznsyp_config.yaml`, set `train_datalist_path`, `valid_datalist_path`, `phn2id_path` and `log_dir`.
Run:
```bash
python train.py -c config/bznsyp_config.yaml
```

### Training for LJSpeech

Edit `config/ljspeech_config.yaml`, set `train_datalist_path`, `valid_datalist_path`, `phn2id_path` and `log_dir`.
Run:
```bash
python train.py -c config/ljspeech_config.yaml
```

## Inference

Edit `inference.ipynb`.
Set `HiFiGAN_CONFIG`, `HiFiGAN_ckpt` and `ckpt_path` to corresponding files, respectively.

* Note: `add_blank` in `inference.ipynb` should be the same as that in `LightGrad/dataset.py`.

## References

* Our model is based on [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones).
* [HiFi-GAN](https://github.com/jik876/hifi-gan) is used as vocoder.
