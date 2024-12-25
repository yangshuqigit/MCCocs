# --------------------------------------------------------
# A script to visualize the clustering results of MCCocs for a given stage, block, head.
# Different layers/heads will present different clustering patterns.
# Refer to Ma, Xu, et al. "Image as set of points." arXiv preprint arXiv:2303.01494 (2023).
# https://arxiv.org/abs/2303.01494

# Use case (generated image will saved to images/cluster_vis/{model}):
# python cluster_visualize.py --image {path_to_image} --model {model} --checkpoint {path_to_checkpoint} --stage {stage} --block {block} --head {head}
# --------------------------------------------------------

import models
import timm
import os
import torch
import argparse
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TransF
from torchvision import transforms
from einops import rearrange
import random
from timm.models import load_checkpoint
from torchvision.utils import draw_segmentation_masks
from nilearn import _utils
from nilearn._utils.niimg import _safe_get_data
import nibabel as nib
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets

object_categories = []
with open("./imagenet_id_to_label_test.txt", "r") as f:
    for line in f:
        _, val = line.strip().split(":")
        object_categories.append(val)

parser = argparse.ArgumentParser(description='Context Cluster visualization')
parser.add_argument('--image', type=str, default="images/A.JPEG", help='path to image')
parser.add_argument('--nii', type=str, default="images/B.nii", help='path to nii')
parser.add_argument('--shape', type=int, default=224, help='image size')
parser.add_argument('--model', default='coc_tiny_plain', type=str, metavar='MODEL', help='Name of model')
parser.add_argument('--stage', default=0, type=int, help='Index of visualized stage, 0-3')
parser.add_argument('--block', default=0, type=int, help='Index of visualized stage, -1 is the last block ,2,3,4,1')
parser.add_argument('--head', default=1, type=int,  help='Index of visualized head, 0-3 or 0-7')
parser.add_argument('--resize_img', action='store_true', default=False, help='Resize img to feature-map size')
parser.add_argument('--checkpoint', type=str, default="coc_tiny_plain.pth.tar", metavar='PATH', help='path to pretrained checkpoint (default: none)')
parser.add_argument('--alpha', default=0.5, type=float, help='Transparent, 0-1')
args = parser.parse_args()
assert args.model in timm.list_models(), "Please use a timm pre-trined model, see timm.list_models()"


# Preprocessing
def _preprocess(nii_path, image_path):
    file_name = nii_path[-8:]
    id = file_name[:-4]
    img_data = load_nifti(nii_path, id)  # load .nii data
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (176,) * 2)
    return img_data, raw_image, id

def Normalization(arr):
    # find min and max
    min_val = np.min(arr)
    max_val = np.max(arr)
    # norm (0, 1)
    return (arr - min_val) / (max_val - min_val)

def load_nifti(file_path, id):
    basc_atlases = datasets.fetch_atlas_aal(version='SPM12', data_dir=None, url=None, resume=True, verbose=1)
    networks_ = basc_atlases.maps
    labels = basc_atlases.labels
    subjects_extractor = NiftiLabelsMasker(
        labels_img=networks_,  # AAL
        memory='nilearn_cache',  # joblib
        memory_level=2,
    )
    # original MRI
    img = nib.load(file_path)
    img_data = img.get_fdata()

    # sMRI
    t1_path = f"G:\\UCLA\\T1ImgNewSegment\\sub_{id}"
    t1_c1_file = get_files_with_prefix(t1_path,'wc1')
    t1_c2_file = get_files_with_prefix(t1_path,'wc2')
    t1_c3_file = get_files_with_prefix(t1_path, 'wc3')
    # white matter
    t1_c1_img = nib.load(os.path.join(t1_path,t1_c1_file[0]))
    resampled_t1_c1 = subjects_extractor._cache(image.resample_img, func_memory_level=2)(
        t1_c1_img, interpolation="nearest",
        target_shape=img.shape[:3],
        target_affine=img.affine)
    t1_c1_data = resampled_t1_c1.get_fdata()
    t1_c1_data = Normalization(t1_c1_data)

    # gray matter
    t1_c2_img = nib.load(os.path.join(t1_path,t1_c2_file[0]))
    resampled_t1_c2 = subjects_extractor._cache(image.resample_img, func_memory_level=2)(
        t1_c2_img, interpolation="nearest",
        target_shape=img.shape[:3],
        target_affine=img.affine)
    t1_c2_data = resampled_t1_c2.get_fdata()
    t1_c2_data = Normalization(t1_c2_data)

    # 脑脊液
    t1_c3_img = nib.load(os.path.join(t1_path,t1_c3_file[0]))
    resampled_t1_c3 = subjects_extractor._cache(image.resample_img, func_memory_level=2)(
        t1_c3_img, interpolation="nearest",
        target_shape=img.shape[:3],
        target_affine=img.affine)
    t1_c3_data = resampled_t1_c3.get_fdata()
    t1_c3_data = Normalization(t1_c3_data)

    # ALFF
    f_ALFF_path = f'ALFF data file path'
    f_ALFF = nib.load(f_ALFF_path)
    f_ALFF_data = f_ALFF.get_fdata()
    f_ALFF_data = Normalization(f_ALFF_data)

    # fALFF
    f_fALFF_path = f'fALFF data file path'
    f_fALFF = nib.load(f_fALFF_path)
    f_fALFF_data = f_fALFF.get_fdata()
    f_fALFF_data = Normalization(f_fALFF_data)

    # ReHo
    f_ReHo_path = f'ReHo data file path'
    f_ReHo = nib.load(f_ReHo_path)
    f_ReHo_data = f_fALFF.get_fdata()
    f_ReHo_data = Normalization(f_ReHo_data)


    # VMHC
    f_VMHC_path = f'VMHC data file path'
    f_VMHC = nib.load(f_VMHC_path)
    f_VMHC_data = f_VMHC.get_fdata()
    f_VMHC_data = Normalization(f_VMHC_data)

    # Degree centrality
    f_CD_path = f'Degree centrality data file path'
    f_CD = nib.load(f_CD_path)
    f_CD_data = f_CD.get_fdata()
    f_CD_data = Normalization(f_CD_data)

    # get mean values
    mean = getVoxelMean(img_data)
    mean = Normalization(mean)
    # 获取标准差
    std = getVoxelStd(img_data)
    std = Normalization(std)

    labels_img_ = _utils.check_niimg_3d(networks_)
    # resampled
    resampled_labels_img_ = subjects_extractor._cache(image.resample_img, func_memory_level=2)(
        labels_img_, interpolation="nearest",
        target_shape=img.shape[:3],
        target_affine=img.affine)
    brain_data = Feature_engineering(img, resampled_labels_img_, f_CD_data, f_ReHo_data, f_ALFF_data, t1_c1_data, t1_c2_data)
    brain_tensor = torch.from_numpy(brain_data)
    brain_tensor = brain_tensor.to(torch.float32)
    return brain_tensor

def getVoxelMean(cal_data):
    return np.mean(cal_data, axis=3)

def getVoxelStd(cal_data):
    return np.std(cal_data, axis=3)

def get_files_with_prefix(folder_path, prefix):
    files = os.listdir(folder_path)
    filtered_files = [file for file in files if file.startswith(prefix)]
    return filtered_files

def Feature_engineering(imgs, labels_img, f1, f2, f3, f4, f5, background_label=0):
    labels_img = _utils.check_niimg_3d(labels_img)
    # TODO: Make a special case for list of strings (load one image at a
    # time).
    target_affine = imgs.affine
    target_shape = imgs.shape[:3]

    # Check shapes and affines.
    if labels_img.shape != target_shape:
        raise ValueError("labels_img and imgs shapes must be identical.")
    if abs(labels_img.affine - target_affine).max() > 1e-9:
        raise ValueError("labels_img and imgs affines must be identical")

    # Perform computation
    labels_data = _safe_get_data(labels_img, ensure_finite=True)
    labels = list(np.unique(labels_data))
    if background_label in labels:
        labels.remove(background_label)
    data = _safe_get_data(imgs, ensure_finite=True)
    cal_data = np.array(data)

    # Obtaining brain representation
    brain_data = np.zeros((116, 1600, 5))
    for index, roi_element in enumerate(labels):
        count = 0
        # single ROI
        roi_data = np.full((1600, 5), -1, dtype = np.float32)
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                for z in range(data.shape[2]):
                    if labels_data[x, y, z] == roi_element:
                        roi_data[count] = [f1[x,y,z], f2[x,y,z], f3[x,y,z], f4[x,y,z], f5[x,y,z]]
                        count += 1
        # set all ROIs
        brain_data[index] = roi_data
    # Brain representation
    brain_data = np.transpose(brain_data, (0,2,1))
    return brain_data


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,M,D]
    :param x2: [B,N,D]
    :return: similarity matrix [B,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    sim = torch.matmul(x1, x2.permute(0, 2, 1))
    return sim


# forward hook function
def get_attention_score(self, input, output):
    x = input[0]  # input tensor in a tuple
    value = self.v(x)
    x = self.f(x)
    x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
    value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
    if self.fold_w > 1 and self.fold_h > 1:
        b0, c0, w0, h0 = x.shape
        assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
            f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
        x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                      f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
        value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
    b, c, w, h = x.shape
    centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
    value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
    b, c, ww, hh = centers.shape
    sim = torch.sigmoid(self.sim_beta +
                        self.sim_alpha * pairwise_cos_sim(
                            centers.reshape(b, c, -1).permute(0, 2, 1),
                            x.reshape(b, c, -1).permute(0, 2,1)
                        )
                    )  # [B,M,N]
    # sololy assign each point to one center
    sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
    mask = torch.zeros_like(sim)  # binary #[B,M,N]
    mask.scatter_(1, sim_max_idx, 1.)  # binary #[B,M,N]
    # changed, for plotting mask.
    mask = mask.reshape(mask.shape[0], mask.shape[1], w, h)  # [(head*fold*fold),m, w,h]
    mask = rearrange(mask, "(h0 f1 f2) m w h -> h0 (f1 f2) m w h",
                     h0=self.heads, f1=self.fold_w, f2=self.fold_h)  # [head, (fold*fold),m, w,h]
    mask_list = []
    for i in range(self.fold_w):
        for j in range(self.fold_h):
            for k in range(mask.shape[2]):
                temp = torch.zeros(self.heads, w * self.fold_w, h * self.fold_h)
                temp[:, i * w:(i + 1) * w, j * h:(j + 1) * h] = mask[:, i * self.fold_w + j, k, :, :]
                mask_list.append(temp.unsqueeze(dim=0))  # [1, heads, w, h]

    mask2 = torch.concat(mask_list, dim=0)  # [ n, heads, w, h]
    global attention
    attention = mask2.detach()


def main():
    global attention
    image, raw_image, id = _preprocess(args.nii, args.image)
    image = image.unsqueeze(dim=0)
    model = timm.create_model(model_name=args.model, pretrained=True)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, True)
        print(f"\n\n==> Loaded checkpoint")
    else:
        print(f"\n\n==> NO checkpoint is loaded")
    model.network[args.stage * 2][args.block].token_mixer.register_forward_hook(get_attention_score)
    out, roi_indices, roi_topk = model(image)
    if type(out) is tuple:
        out = out[0]
    possibility = torch.softmax(out, dim=1).max()
    value, index = torch.max(out, dim=1)
    print(f'==> Prediction is: {object_categories[index]} possibility: {possibility * 100:.3f}%')
    print(f'idx = {roi_indices}\ntopk = {roi_topk}')

    try:
        os.makedirs(f"images/cluster_vis/{args.model}")
    except:
        pass

    image_name = os.path.basename(args.image).split(".")[0]

    from torchvision.io import read_image
    img = read_image(args.image)
    # process the attention map
    attention = attention[:, args.head, :, :]
    mask = attention.unsqueeze(dim=0)
    mask = F.interpolate(mask, (img.shape[-2], img.shape[-1]))
    mask = mask.squeeze(dim=0)
    mask = mask > 0.5
    # randomly selected some good colors.
    colors = [
        # Enter a list of colors, such as rgb or color names
    ]
    if mask.shape[0] == 11:
        colors = colors[0:11]
    if mask.shape[0] > 11:
        colors = colors * (mask.shape[0] // 121)
        random.seed(42)
        random.shuffle(colors)

    img_with_masks = draw_segmentation_masks(img, masks=mask, alpha=args.alpha, colors=colors)
    img_with_masks = img_with_masks.detach()
    img_with_masks = TransF.to_pil_image(img_with_masks)
    img_with_masks = np.asarray(img_with_masks)
    save_path = f"images/cluster_vis/{args.model}/{id}_Stage{args.stage}_Block{args.block}_Head{args.head}.png"
    cv2.imwrite(save_path, img_with_masks)
    print(f"==> Generated image is saved to: {save_path}")


if __name__ == '__main__':
    main()
