import torch
from glob import glob
import argparse
import numpy as np
import os
from PIL import Image
from gmflow.gmflow import GMFlow
from gmflow_utils.frame_utils import read_gen
from gmflow_evaluate import inference_on_dir, FlowPredictor

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint', default='', type=str,
                        help='where to load the checkpoint for the model')
    parser.add_argument('--stage', default='chairs', type=str,
                        help='training stage')
    parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                        help='image size for training')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')

    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                        help='validation dataset')
    parser.add_argument('--with_speed_metric', action='store_true',
                        help='with speed metric when evaluation')

    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # GMFlow model
    parser.add_argument('--num_scales', default=1, type=int,
                        help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)

    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    # inference on a directory
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
    parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')

    parser.add_argument('--output_dir', '-o', default='output', type=str,
                        help='where to save the prediction results')
    parser.add_argument('--save_raw', action='store_true')

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time on sintel')
    
    parser.add_argument('--finger_mask', '-m', default='tmp', type=str,
                        help='where to load finger mask')

    return parser


def main(args):

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:{}'.format(args.local_rank))
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    finger_mask_tensor = torch.tensor(np.asarray(Image.open(args.finger_mask)),
                                   device=device)
    finger_mask = finger_mask_tensor[:,:,0] > 0

    
    # model
    model = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model_without_ddp = model.module
    else:
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)

            model_without_ddp = model.module
        else:
            model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
    
    # resume checkpoints
    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        loc = 'cuda:{}'.format(args.local_rank)
        checkpoint = torch.load(args.resume, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model_without_ddp.load_state_dict(weights, strict=args.strict_resume)







    # model
    model = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
    
    flow_predictor = FlowPredictor(model, 
                                   args.attn_splits_list,
                                   args.corr_radius_list,
                                   args.prop_radius_list,)
    
    # resume checkpoints
    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        loc = 'cuda:{}'.format(args.local_rank)
        checkpoint = torch.load(args.resume, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model.load_state_dict(weights, strict=args.strict_resume)

    # # resume checkpoints
    # if args.checkpoint != '':
    #     print('Load checkpoint: %s' % args.checkpoint)

    #     loc = 'cuda:{}'.format(args.local_rank)
    #     checkpoint = torch.load(args.resume, map_location=loc)
    #     weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    #     model.load_state_dict(weights, strict=args.strict_resume)

    flow_mag_dir = os.path.join(args.output_dir, 'flow_mag_pred')
    flow_rad_dir = os.path.join(args.output_dir, 'flow_rad_pred')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(flow_mag_dir, exist_ok=True)
    os.makedirs(flow_rad_dir, exist_ok=True)
    if args.save_raw:
        raw_dir = os.path.join(args.output_dir, 'flow_raw_pred') 
        os.makedirs(raw_dir, exist_ok=True)

    filenames = sorted(glob(args.inference_dir + '/*'))
    print('%d images found' % len(filenames))

    for test_id in range(len(filenames) - 1):

        image1 = read_gen(filenames[test_id])
        image2 = read_gen(filenames[test_id + 1])

        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        if len(image1.shape) == 2:  # gray image, for example, HD1K
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

        flow_predictor.pred(image1, image2, finger_mask)




if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
