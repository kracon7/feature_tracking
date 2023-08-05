import torch
import torchvision.transforms as T
import cv2
import numpy as np
import argparse
import time
import torch

class FeatureExtractor:
    def __init__(self, dino_model, img) -> None:
        '''
        Args:
            dino_model -- DINO v2 model
            img -- (numpy.ndarray) shape (w, h, 3)
        '''
        self.dino_model = dino_model
        self.dino_transform = T.Compose([
                T.Normalize(mean=[0.5], std=[0.5]),
                ])
        h, w = img.shape[0], img.shape[1]
        self.crop_h = (h - h%14) - 14
        self.crop_w = (w - w%14) - 14
        self.patch_h, self.patch_w = self.crop_h // 14, self.crop_w // 14

        self.offset = np.stack(np.meshgrid(np.linspace(0, 15, 6), 
                                    np.linspace(0, 15, 6),
                                    indexing='ij'),
                        axis=-1).reshape(-1, 2).astype('uint8')
        self.batch_size = self.offset.shape[0]
        self.grid_size = int(np.sqrt(self.batch_size))
        

    def augment_img(self, img):
        '''
        Crop and shift image every 3 pixels
        '''
        aug_img = np.zeros((36, self.crop_h, self.crop_w, 3))
        for i, (dh, dw) in enumerate(self.offset):
            aug_img[i] = img[dh:self.crop_h+dh, dw:self.crop_w+dw]

        aug_img = torch.tensor(aug_img.transpose(0, 3, 1, 2)).cuda()
        return aug_img

    def extract_feature(self, img):
        img_tensor = self.augment_img(img)
        dino_im = self.dino_transform(img_tensor).float().cuda()
        with torch.no_grad():
            ret = self.dino_model.forward_features(dino_im)
            patch_tok = ret["x_norm_patchtokens"].view(self.batch_size, 
                                                       self.patch_h, 
                                                       self.patch_w, 
                                                       -1)
        return patch_tok


def main(args):
    cap = cv2.VideoCapture(args.video)
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()

    ret, first_frame = cap.read()
    feature_extractor = FeatureExtractor(dinov2_model, first_frame)

    # Extract features in the first frame
    initial_feature = feature_extractor.extract_feature(first_frame)

    for i in range(20):
        now = time.time()
        _, frame = cap.read()
        feature = feature_extractor.extract_feature(frame)

        print("Feature extraction takes: %.2fms"%(1e3 * (time.time() - now)))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=str,
                        default="")

    args = parser.parse_args()
    main(args)