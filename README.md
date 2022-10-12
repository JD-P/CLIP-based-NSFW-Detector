# CLIP-based-NSFW-Detector

This 2 class NSFW-detector is a lightweight Autokeras model that takes CLIP ViT L/14 embbedings as inputs.
It estimates a value between 0 and 1 (1 = NSFW) and works well with embbedings from images.

DEMO-Colab:
https://colab.research.google.com/drive/19Acr4grlk5oQws7BHTqNIK-80XGw2u8Z?usp=sharing

The training CLIP V L/14 embbedings can be downloaded here:
https://drive.google.com/file/d/1yenil0R4GqmTOFQ_GVw__x61ofZ-OBcS/view?usp=sharing (not fully manually annotated so cannot be used as test)


The (manually annotated) test set is there https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/nsfw_testset.zip

https://github.com/rom1504/embedding-reader/blob/main/examples/inference_example.py inference on laion5B

## Installation

Run the following commands to install the dependencies

```
python3 -m venv env_nsfw
source env_nsfw/bin/activate
pip install clip-anytorch autokeras
git clone https://github.com/openai/CLIP/
pip install ./CLIP
```

You may have to reinstall pytorch with the right CUDA for your device, for most datacenter GPUs the command is:

```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Then run the `find_nsfw.py` demo utility on a flat directory of images:

```
python3 find_nsfw.py DIRECTORY
```
