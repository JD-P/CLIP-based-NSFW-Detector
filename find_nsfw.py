"""Find NSFW files in a directory of images."""

import os
from argparse import ArgumentParser
from PIL import Image
from CLIP import clip
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch
import numpy as np
import autokeras as ak
from tensorflow.keras.models import load_model
from urllib.request import urlretrieve
import zipfile
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("image_dir", help="The directory of images to classify.")
parser.add_argument("-o", "--outfile", type=str, default="nsfw_images.txt",
                    help="The file to write NSFW paths to.")
parser.add_argument("--threshold", type=float, default=0.1,
                    help="The value over which to print a filepath")
parser.add_argument("--batch-size", type=int, default=1024,
                    help="How many images to process per batch.")
args = parser.parse_args()

def load_nsfw_detector(clip_model):
    """load the NSFW detection model"""
    
    cache_folder = "./NSFW-cache"

    if clip_model == "ViT-L/14":
        model_dir = cache_folder + "/clip_autokeras_binary_nsfw"
        dim = 768
    elif clip_model == "ViT-B/32":
        model_dir = cache_folder + "/clip_autokeras_nsfw_b32"
        dim = 512
    else:
        raise ValueError("Unknown clip model")
    if not os.path.exists(model_dir):
        os.makedirs(cache_folder, exist_ok=True)

        path_to_zip_file = cache_folder + "/clip_autokeras_binary_nsfw.zip"
        if clip_model == "ViT-L/14":
            url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        elif clip_model == "ViT-B/32":
            url_model = (
                "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_nsfw_b32.zip"
            )
        else:
            raise ValueError("Unknown model {}".format(clip_model))  # pylint: disable=consider-using-f-string
        urlretrieve(url_model, path_to_zip_file)

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)

    loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    loaded_model.predict(np.random.rand(10**3, dim).astype("float32"), batch_size=10**3)

    return loaded_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
clip_model_name = 'ViT-L/14'
clip_model= clip.load(clip_model_name, jit=False, device=device)[0]
clip_model.eval().requires_grad_(False)

safety_model = load_nsfw_detector("ViT-L/14")

side_x = 224
side_y = 224

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

# TODO: Optimize this to use batches?
outfile = open(args.outfile, "w")
partial_paths = os.listdir(args.image_dir)
for pp_batch_i in tqdm(range(0,
                             len(partial_paths) + args.batch_size,
                             args.batch_size)):
    image_paths = []
    images = []
    for partial_path in partial_paths[pp_batch_i:pp_batch_i + args.batch_size]:
        fullpath = os.path.join(args.image_dir, partial_path)
        img = Image.open(fullpath).convert('RGB')
        img = TF.resize(img, min(side_x, side_y, *img.size),
                        transforms.InterpolationMode.LANCZOS)
        img = TF.center_crop(img, (side_x, side_y)[::-1])
        img = TF.to_tensor(img).to(device)
        image_paths.append(fullpath)
        images.append(img)
    if not images:
        continue
    images = torch.stack(images).to(device)
    embed = np.asarray(
        normalized(
            clip_model.encode_image(images).cpu()
        )
    )
    nsfw_scores = safety_model.predict(embed)
    for i in range(len(nsfw_scores)):       
        if nsfw_scores[i] >= args.threshold:
            outfile.write(f"{image_paths[i]}:{nsfw_scores[i]}\n")
            outfile.flush()
outfile.close()
