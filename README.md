# megatts2
Unofficial implementation of Megatts2

## TODO
### Base test
- [x] Prepare dataset
- [x] VQ-GAN
- [x] ADM
- [x] PLM
### Better version
- [ ] Replace Hifigan with Bigvgan
- [ ] Mix training Chinese and English
- [ ] Train on about 1k hours of speech
- [ ] Webui

## colab
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

export PATH=$PATH:/root/miniconda3/bin

## Install mfa
1. conda create -n aligner && conda activate aligner
2. conda install -c conda-forge montreal-forced-aligner=2.2.17

wget https://us.openslr.org/resources/60/test-clean.tar.gz
tar xzf test-clean.tar.gz

cd drive/MyDrive/megatts2/   

pip install pypinyin
pip install phonemizer
pip install lhotse
pip install speechbrain
pip install -r requirements.txt


export HF_ENDPOINT=https://hf-mirror.com
export ONEDNN_PRIMITIVE_CACHE_CAPACITY=0
export LRU_CACHE_CAPACITY=1
export TORCH_CUDNN_V8_API_DISABLED=1
   
## Prepare dataset
1. Prepare wav and txt files to ./data/wav 
2. Run `python3 prepare_ds.py --stage 0 --num_workers 4 --wavtxt_path /data/sky/data/wavs --text_grid_path /data/sky/data/textgrids --ds_path /data/sky/data/ds`
3. mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
4. mfa align /data/sky/data/wavs english_mfa english_mfa /data/sky/data/textgrids --clean -j 12 -t tmp
5. Run `python3 prepare_ds.py --stage 1 --num_workers 4 --wavtxt_path /data/sky/data/wavs --text_grid_path /data/sky/data/textgrids --ds_path /data/sky/data/ds` 
6. Run `python3 prepare_ds.py --stage 2 --generator_config configs/config_gan.yaml --generator_ckpt generator.ckpt` after training generator.

## Train
Training procedure refers to Pytorch-lightning
python3 cli.py fit --config configs/config_adm.yaml 

## Infer test
`python infer.py`

## License
- MIT
- Support by Simon of [ZideAI](https://zideai.com/)
