import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
import argparse
import time
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        rgba = self.scalarMap.to_rgba(val)
        rgb = (int(255*rgba[0]), int(255*rgba[1]), int(255*rgba[2]))
        return rgb

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
        
        # Mask for efficient densify image sampling
        self.offset = torch.stack(torch.meshgrid(torch.arange(0, 14, self.stride, device=device), 
                                                 torch.arange(0, 14, self.stride, device=device),
                                                 indexing='ij'),
                                    dim=-1).reshape(-1, 2)
        self.batch_size = self.offset.shape[0]
        self.grid_size = int(np.sqrt(self.batch_size))
        self.densify_mask = torch.zeros((self.batch_size, 3, h, w), dtype=torch.bool, device=device)
        for i, (oh, ow) in enumerate(self.offset):
            self.densify_mask[i, :, oh:self.crop_h+oh, ow:self.crop_w+ow] = 1        

        # Search window size
        self.search_range = 10
        self.win_size = 2 * self.search_range + 1

        # Use flattened array of embedings for efficient slicing
        window_locs = torch.zeros([self.dpatch_h, self.dpatch_w, self.win_size, self.win_size, 2], 
                                  dtype=torch.long, device=device)
        for i in range(self.dpatch_h):
            for j in range(self.dpatch_w):
                window_locs[i,j] = torch.stack(
                            torch.meshgrid(torch.arange(i, i + self.win_size, 1, device=device), 
                                           torch.arange(j, j + self.win_size, 1, device=device),
                                                 indexing='ij'), dim=-1)
        window_locs = window_locs.reshape(self.dpatch_h * self.dpatch_w, 
                                                 self.win_size * self.win_size, 
                                                 2)
        self.flat_window_idxs = window_locs[:,:,0] * (self.dpatch_w + 2 * self.search_range)\
                                + window_locs[:,:,1]


    def densify(self, img):
        '''
        Crop and shift image every k pixels, k = self.stride
        '''
        # Normalize and move to cuda
        img_tensor = torch.tensor(img).permute([2,0,1]).float().cuda() / 255
        dino_im = self.dino_transform(img_tensor)

        # Repeat by number of batches
        dino_im = dino_im.unsqueeze(0).repeat(self.batch_size, 1, 1 ,1)

        # Apply mask
        aug_img = dino_im[self.densify_mask].view(self.batch_size, 3, self.crop_h, self.crop_w)

        return aug_img

    def extract_feature(self, img):
        dino_im = self.densify(img)
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

    def calculate_flow(self, old_embedings, new_embedings, old_locs, threshold=0.95, confidence=1.2):
        '''
        Window search between old_embedings and new_embedings at locations
        defined by locs.
        Args:
            old_embedings -- (torch.tensor) embedings from the previous frame
            new_embedings -- (torch.tensor) embedings from the next frame
            locs -- (torch.tensor) n x 2 locations
        '''
        num_old_locs = old_locs.shape[0]
        old_embedings = old_embedings[old_locs[:,0], old_locs[:,1]].unsqueeze(1)
        pad = [0,0, self.search_range, self.search_range, self.search_range, self.search_range]
        padded_new_embedings = F.pad(new_embedings, pad, mode='constant', value=0)
        flat_padded_new_embedings = padded_new_embedings.reshape(-1, self.embed_dim)

        flat_old_locs = old_locs[:,0] * self.dpatch_w + old_locs[:,1]
        flat_new_window_idxs = self.flat_window_idxs[flat_old_locs].reshape(-1)
        new_embedings_windows = torch.index_select(flat_padded_new_embedings,
                                                   dim=0,
                                                   index=flat_new_window_idxs)\
                                    .reshape(num_old_locs, self.win_size**2, self.embed_dim)

        # Cos similarity between one feature and window
        similarity = self.cossim(old_embedings.clone(), new_embedings_windows.clone())
        top_similarity, match_idx = torch.max(similarity, dim=1)
        match_idx = torch.argmax(similarity, dim=1)
        shift = torch.stack([match_idx // self.win_size - self.search_range, 
                             match_idx % self.win_size - self.search_range]).T
        new_locs = old_locs + shift
        status = top_similarity > threshold

        # Confidence filter
        ave_similarity = torch.mean(similarity, dim=1)
        status = status & ((top_similarity / ave_similarity) > confidence)

        return new_locs, status
    
    def locs_to_pixels(self, locs):
        '''
        Convert coordinates in embeding axis to image pixels
        Args:
            locs -- (np.ndarray) n x 2
        '''
        pixels = locs * self.stride + 7
        return pixels


def main(args):
    mpl_color_helper = MplColorHelper('binary', 0, 20)

    cap = cv2.VideoCapture(args.video)
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
    dinov2_encoder = DINOV2Encoder(dinov2_model)
    embed_dim = dinov2_model.embed_dim
    if args.parallel:
        dinov2_encoder = torch.nn.DataParallel(dinov2_encoder)

    # Skip the first few seconds
    for i in range(150):
        ret, prev_frame = cap.read()
    
    feature_extractor = FeatureExtractor(dinov2_encoder, prev_frame, embed_dim)

    # Extract features in the first frame
    old_embedings = feature_extractor.extract_feature(prev_frame)

    locs = torch.stack(torch.meshgrid(torch.arange(0, feature_extractor.dpatch_h, 10),
                                      torch.arange(0, feature_extractor.dpatch_w, 10),
                                      indexing='ij'),
                            dim=-1).reshape(-1, 2).cuda()

    for i in range(1000):
        now = time.time()
        _, frame = cap.read()
        new_embdeings = feature_extractor.extract_feature(frame)

        print("Feature extraction takes: %.2fms"%(1e3 * (time.time() - now)))
        
        now = time.time()
        new_locs, status = feature_extractor.calculate_flow(old_embedings, 
                                                            new_embdeings,
                                                            locs)
        print("Feature matching takes: %.2fms"%(1e3 * (time.time() - now)))

        canvas = np.zeros_like(prev_frame)
        valid_start_pixels = feature_extractor.locs_to_pixels(locs)[status].cpu().numpy()
        valid_end_pixels = feature_extractor.locs_to_pixels(new_locs)[status].cpu().numpy()
        for start, end in zip(valid_start_pixels, valid_end_pixels):
            color = mpl_color_helper.get_rgb(np.linalg.norm(start-end))
            end = start + 2 * (end - start)
            canvas = cv2.arrowedLine(canvas, 
                                     (start[1], start[0]), 
                                     (end[1], end[0]),
                                     color,
                                     1,
                                     tipLength=0.2)
        result = cv2.add(prev_frame, canvas)
        cv2.imshow('Optical Flow (sparse)', result)
        cv2.waitKey(1)

        prev_frame = frame
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