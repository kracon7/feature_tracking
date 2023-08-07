import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
import argparse
import time
import torch

class DINOV2Encoder(nn.Module):
    def __init__(self, dinov2_model):
        super(DINOV2Encoder, self).__init__()
        self.dinov2_model = dinov2_model

    def forward(self, input):
        ret = self.dinov2_model.forward_features(input)
        patch_tok = ret["x_norm_patchtokens"]
        return patch_tok

class FeatureExtractor:
    def __init__(self, encoder, img, embed_dim, device='cuda') -> None:
        '''
        Args:
            dino_model -- DINO v2 model
            img -- (numpy.ndarray) shape (w, h, 3)
        '''
        self.encoder = encoder
        self.device = device
        self.embed_dim = embed_dim
        self.dino_transform = T.Compose([
                T.Normalize(mean=[0.5], std=[0.5]),
                ])
        self.cossim = nn.CosineSimilarity(dim=-1, eps=1e-6)

        h, w = img.shape[0], img.shape[1]
        self.crop_h = (h - h%14) - 14
        self.crop_w = (w - w%14) - 14
        self.stride = 2
        self.patch_h, self.patch_w = self.crop_h // 14, self.crop_w // 14
        # Densified  embeding shape
        self.dpatch_h, self.dpatch_w = self.crop_h // self.stride, self.crop_w // self.stride
        
        self.offset = np.stack(np.meshgrid(np.arange(0, 14, self.stride), 
                                    np.arange(0, 14, self.stride),
                                    indexing='ij'),
                        axis=-1).reshape(-1, 2).astype('int')
        self.batch_size = self.offset.shape[0]
        self.grid_size = int(np.sqrt(self.batch_size))

        # Search window size
        self.win_size = 10

    def densify(self, img):
        '''
        Crop and shift image every 3 pixels
        '''
        aug_img = np.zeros((self.batch_size, self.crop_h, self.crop_w, 3))
        for i, (dh, dw) in enumerate(self.offset):
            aug_img[i] = img[dh:self.crop_h+dh, dw:self.crop_w+dw]

        aug_img = torch.tensor(aug_img.transpose(0, 3, 1, 2)).cuda()
        return aug_img

    def extract_feature(self, img):
        img_tensor = self.densify(img)
        dino_im = self.dino_transform(img_tensor).float().cuda()
        with torch.no_grad():
            embedings = self.encoder(dino_im).view(self.batch_size, 
                                                    self.patch_h, 
                                                    self.patch_w, 
                                                    self.embed_dim)
        embedings = embedings.permute([1,2,0,3]).reshape(
                self.patch_h, self.patch_w, self.grid_size, self.grid_size, self.embed_dim)
        embedings = torch.cat([torch.cat(list(embedings[i]), dim=1) 
                             for i in range(self.patch_h)], 
                             dim=0)
        return embedings

    def calculate_flow(self, old_embedings, new_embedings, old_locs, threshold=0.9):
        '''
        Window search between old_embedings and new_embedings at locations
        defined by locs.
        Args:
            old_embedings -- (torch.tensor) embedings from the previous frame
            new_embedings -- (torch.tensor) embedings from the next frame
            locs -- (np.ndarray) n x 2 locations
        '''
        old_embedings = old_embedings[old_locs[:,0], old_locs[:,1]].unsqueeze(1)
        pad = [0,0, self.win_size, self.win_size, self.win_size, self.win_size]
        padded_new_embedings = F.pad(new_embedings, pad, mode='constant', value=0)
        new_embedings_windows = []
        for i, j in old_locs:
            window = padded_new_embedings[i:i+2*self.win_size+1, 
                                          j:j+2*self.win_size+1]
            new_embedings_windows.append(window.reshape((-1, self.embed_dim)))
        new_embedings_windows = torch.stack(new_embedings_windows)
        # Cos similarity between one feature and window
        similarity = self.cossim(old_embedings, new_embedings_windows)
        top_similarity, match_idx = torch.max(similarity, dim=1)
        match_idx = torch.argmax(similarity, dim=1)
        shift = torch.stack([match_idx // (2*self.win_size+1), 
                             match_idx % (2*self.win_size+1)]).T
        new_locs = old_locs + shift.cpu().numpy()
        status = top_similarity > threshold
        return new_locs, status


def main(args):
    cap = cv2.VideoCapture(args.video)
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
    dinov2_encoder = DINOV2Encoder(dinov2_model)
    embed_dim = dinov2_model.embed_dim
    if args.parallel:
        dinov2_encoder = torch.nn.DataParallel(dinov2_encoder)

    ret, first_frame = cap.read()
    feature_extractor = FeatureExtractor(dinov2_encoder, first_frame, embed_dim)

    # Extract features in the first frame
    old_embedings = feature_extractor.extract_feature(first_frame)

    locs = np.stack(np.meshgrid(np.arange(0, feature_extractor.dpatch_h, 10), 
                                    np.arange(0, feature_extractor.dpatch_w, 10),
                                    indexing='ij'),
                        axis=-1).reshape(-1, 2).astype('int')

    for i in range(20):
        now = time.time()
        _, frame = cap.read()
        new_embdeings = feature_extractor.extract_feature(frame)

        print("Feature extraction takes: %.2fms"%(1e3 * (time.time() - now)))
        
        now = time.time()
        new_locs, status = feature_extractor.calculate_flow(old_embedings, 
                                                            new_embdeings,
                                                            locs)
        print("Feature matching takes: %.2fms"%(1e3 * (time.time() - now)))
        old_embedings = new_embdeings



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--parallel', action="store_true", default=False)
    parser.add_argument("--video", type=str,
                        default="")

    args = parser.parse_args()
    main(args)