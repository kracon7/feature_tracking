import torch
import torchvision.transforms as T
import cv2
import numpy as np
import argparse
import time
import torch

def augment_img(img):
    '''
    Crop and shift image every 3 pixels
    '''
    h, w = img.shape[0], img.shape[1]
    h = (h - h%14) - 14
    w = (w - w%14) - 14

    offset = np.stack(np.meshgrid(np.linspace(0, 15, 6), 
                                  np.linspace(0, 15, 6),
                                  indexing='ij'),
                     axis=-1).reshape(-1, 2).astype('uint8')

    aug_img = np.zeros((36, h, w, 3))
    for i, (dh, dw) in enumerate(offset):
        aug_img[i] = img[dh:h+dh, dw:w+dw]

    aug_img = torch.tensor(aug_img.transpose(0, 3, 1, 2)).cuda()
    return aug_img


def main(args):
    cap = cv2.VideoCapture(args.video)
    dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()

    ret, first_frame = cap.read()

    img_tensor = augment_img(first_frame)

    dino_transform = T.Compose([
            T.Normalize(mean=[0.5], std=[0.5]),
            ])

    for i in range(20):
        now = time.time()
        dino_im = dino_transform(img_tensor).float().cuda()
        ret = dinov2_vitb14.forward_features(dino_im)
        print("Feature extraction takes: %.2fms"%(1e3 * (time.time() - now)))
        
    # patch_tok = ret["x_norm_patchtokens"]
    # cls_tok = ret["x_norm_clstoken"]

    # _, _, img_h, img_w = dino_im.shape
    # patch_h, patch_w = img_h // 14, img_w // 14

    # patch_tok = einops.rearrange(patch_tok, "b (h w) c -> b c h w", h=patch_h)
    # cls_tok = einops.repeat(cls_tok, "b c -> b c h w", h=patch_h, w=patch_w)
    # dino_features = torch.cat((patch_tok, cls_tok), dim=1)[0].permute(1,2,0).detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=str,
                        default="")

    args = parser.parse_args()
    main(args)