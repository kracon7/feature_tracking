import torch
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import time
from PIL import Image

cudnn.benchmark = False

im_size = (462, 616)
image_pil = Image.open('/home/jiacheng/tmp/rs_test.png')

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()
dino_transform = T.Compose([
        T.Resize(im_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
        ])

for i in range(20):
    now = time.time()
    dino_im = dino_transform(image_pil).unsqueeze(0).cuda()
    dino_im = dino_im.repeat((36,1,1,1))
    with torch.no_grad():
        ret = dinov2_vits14.forward_features(dino_im)
    print("Feature extraction takes: %.2fms"%(1e3 * (time.time() - now)))
    