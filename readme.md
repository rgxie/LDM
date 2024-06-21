
## LDM: Large Tensorial SDF Model for Textured Mesh Generation

This is the official implementation of *LDM: Large Tensorial SDF Model for Textured Mesh Generation*.

### <a href="https://arxiv.org/abs/2405.14580"><img src="https://img.shields.io/badge/ArXiv-2404.07191-brightgreen"></a> | <a href="https://huggingface.co/spaces/rgxie/LDM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a> | [Weights](https://huggingface.co/rgxie/LDM)

https://github.com/rgxie/tensor23d/assets/38643138/4ba879ab-bb52-4c97-9fb0-7e395172f67b


### Features and Todo List
- [x] 🔥 Release huggingface gradio demo
- [x] 🔥 Release inference and training code.
- [x] 🔥 Release pretrained models.
- [x] Release the training data list.
- [x] Support text to 3D generation.
- [x] Support image to 3D generation using various multi-view diffusion models, including Imagedream and Zero123plus.

### Install

```bash
# xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
# We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

# other dependencies
pip install -r requirements.txt
```

### Pretrained Weights

Our pretrained weight can be downloaded from [huggingface](https://huggingface.co/rgxie/LDM).

For example, to download the fp16 model for inference:
```bash
mkdir pretrained && cd pretrained
wget wget https://huggingface.co/rgxie/LDM/resolve/main/LDM_6V_SDF.ckpt
cd ..
```

The weights of the diffusion model will be downloaded automatically.

### Inference


```bash
### gradio app for both text/image to 3D, the weights of our model will be downloaded automatically.
python app.py

# image to 3d
# --workspace: folder to save output (*.obj,*.jpg)
# --test_path: path to a folder containing images, or a single image
python infer.py tiny_trf_trans_sdf --resume pretrained/LDM_6V_SDF.ckpt --workspace workspace_test --test_path example --seed 0

# text to 3d
# --workspace: folder to save output (*.obj,*.jpg)
python infer.py tiny_trf_trans_sdf --resume pretrained/LDM_6V_SDF.ckpt --workspace workspace_test --txt_or_image True --mvdream_or_zero123 True --text_prompt 'a hamburge' --seed 0

```
For more options, please check [options](./core/options.py). If you find the output unsatisfying, try using different multi-view diffusion models or seeds!
### Training

**preparing**: 

Training dataset: our training dataset is based on [GObjaverse](https://aigc3d.github.io/gobjaverse/), which can be downloaded from [here](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse).
Specifically, we used a ~80K filtered subset list from [LGM](https://github.com/3DTopia/LGM). The data list can be found [here](https://github.com/ashawkey/objaverse_filter/blob/main/gobj_merged.json). Furthermore, configure the [options](./core/options.py) with the following:

- data_path: The directory where your downloaded dataset is stored.
- data_debug_list: The path to the data list file.
- The structure of dataset:
```
|-- data_path
    |-- dictionary_id
        |-- instance_id.rar    
        |-- ...
```

Pretrained model: As our model is trained starting from the pretrained [OpenLRM](https://github.com/3DTopia/OpenLRM) model, please download the pretrained model [here](https://huggingface.co/zxhezexin/openlrm-mix-large-1.1/resolve/main/model.safetensors) and place it in the ‘pretrained’ dir.

**Training**:
The minimum recommended configuration for training is 8 * A6000 GPUs, each with 48GB memory.

```bash


# step 1: To speed up the convergence of training, we start by not cropping patches. Instead, we use a lower resolution and train with a larger batch size initially.
accelerate launch --config_file acc_configs/gpu8.yaml main.py tiny_trf_trans_sdf --output_size 64 --batch_size 4 --lr 4e-4 --num_epochs 50 --is_crop False --resume pretrained/openlrm_m_l.safetensors --workspace workspace_nocrop


# step 2: Furthermore, we introduce patch cropping and increase the patch resolution to capture better details.
accelerate launch --config_file acc_configs/gpu8.yaml main.py tiny_trf_trans_sdf --output_size 128 --batch_size 1 --gradient_accumulation_steps 2 --lr 2e-5 --num_epochs 50 --is_crop True --resume workspace_nocrop/last.ckpt --workspace workspace_crop

# (optional)step 3: To adapt the model to the 6 view inputs from Zero123plus, we refine the model obtained in the earlier stages.
accelerate launch --config_file acc_configs/gpu8.yaml main.py tiny_trf_trans_sdf_123plus --output_size 128 --batch_size 1 --gradient_accumulation_steps 2 --lr 1e-5 --num_epochs 20 --resume workspace_crop/last.ckpt --workspace workspace_refine


# (optional)step 4: Utilize FlexiCubes layer to further improve the texture details
accelerate launch --config_file acc_configs/gpu8.yaml main.py tiny_trf_trans_mesh --output_size 512 --batch_size 1 --gradient_accumulation_steps 1 --lr 1e-5 --num_epochs 20 --resume the_path_of_sdf_ckpt/last.ckpt --workspace workspace_mesh

```

### Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [LGM](https://github.com/3DTopia/LGM/tree/main) 
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes)
- [InstantMesh](https://github.com/TencentARC/InstantMesh)

### Citation

```
@article{xie2024ldm,
  title={LDM: Large Tensorial SDF Model for Textured Mesh Generation},
  author={Xie, Rengan and Zheng, Wenting and Huang, Kai and Chen, Yizheng and Wang, Qi and Ye, Qi and Chen, Wei and Huo, Yuchi},
  journal={arXiv preprint arXiv:2405.14580},
  year={2024}
}
```
