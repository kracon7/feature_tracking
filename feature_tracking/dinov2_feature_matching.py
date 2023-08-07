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
        if next(dino_model.parameters()).is_cuda:
            self.device = 'cuda'
        else:
            raise Exception("DINO V2 model is not on GPU!!")
        self.embed_dim = dino_model.embed_dim
        self.dino_transform = T.Compose([
                T.Normalize(mean=[0.5], std=[0.5]),
                ])
        h, w = img.shape[0], img.shape[1]
        self.crop_h = (h - h%14) - 14
        self.crop_w = (w - w%14) - 14
        self.stride = 2
        self.patch_h, self.patch_w = self.crop_h // 14, self.crop_w // 14
        self.dense_patch_h, self.dense_patch_w = self.crop_h // self.stride, self.crop_w // self.stride
        
        self.offset = np.stack(np.meshgrid(np.arange(0, 14, self.stride), 
                                    np.arange(0, 14, self.stride),
                                    indexing='ij'),
                        axis=-1).reshape(-1, 2).astype('uint8')
        self.batch_size = self.offset.shape[0]
        self.grid_size = int(np.sqrt(self.batch_size))

        # Search window size
        self.win_size = 10
        self.roll_mask = torch.zeros((self.dense_patch_h, 
                                      self.dense_patch_w,
                                      self.win_size + 1,
                                      self.win_size + 1,
                                      self.embed_dim)).bool().to(self.device)
        self.roll_shifts = []
        for i in range(-self.win_size, self.win_size+1, 1):
            for j in range(-self.win_size, self.win_size+1, 1):
                self.roll_shifts.append((i, j))
                if i > 0:
                    self.roll_mask[:i,:,i,j,:] = 1
                elif i < 0:
                    self.roll_mask[i:,:,i,j,:] = 1

                if j > 0:
                    self.roll_mask[:,:j,i,j,:] = 1
                elif j < 0:
                    self.roll_mask[:,j:,i,j,:] = 1

    def augment_img(self, img):
        '''
        Crop and shift image every 3 pixels
        '''
        aug_img = np.zeros((self.batch_size, self.crop_h, self.crop_w, 3))
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
                                                       self.embed_dim)
        patch_tok = patch_tok.permute([1,2,0,3]).reshape(
                self.patch_h, self.patch_w, self.grid_size, self.grid_size, self.embed_dim)
        feature = torch.cat([torch.cat(list(patch_tok[i]), dim=1) 
                             for i in range(self.patch_h)], 
                             dim=0)
        return feature

    def roll_feature(self, feature):
        '''
        Arg:
            feature -- (torch.tensor) shape hf x wf x df
        Return:
            rolled -- (torch.tensor) shape hf x wf x hr x wr x df
        '''
        rolled = torch.zeros(self.roll_mask.shape).float().to(self.device)
        for i, j in self.roll_shifts:
            rolled[:,:, i,j] = torch.roll(feature, (i,j), (0,1))
        
        rolled[self.roll_mask] = 0
        return rolled

    def patch_idx_to_image_idx(self, patch_idx):
        pass


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

        rolled = feature_extractor.roll_feature(feature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video", type=str,
                        default="")

    args = parser.parse_args()
    main(args)