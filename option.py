import argparse
import template

# from options import merge_duf_mixs2_opt

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")
parser.add_argument('--exp_name', type=str, default="HSIMamba", help="name of experiment")
parser.add_argument('--template', default='HQSManba',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/Data/', help='dataset directory')
parser.add_argument('--train_root',type=str,default="/root//autodl-tmp/Data/")
# Saving specifications
parser.add_argument('--outf', type=str, default='/root/autodl-tmp/Model/HQSManba_1103_newtype/', help='saving_path')

# Model specifications autodl-tmp/Data/TSA_simu_data/mask_3d_shift.mat
parser.add_argument('--method', type=str, default='HQSManbaOri_2', help='method name')
parser.add_argument('--shared', type=bool, default=True)
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument('--resume_ckpt_path', type=str, default=None, help='resumed checkpoint directory')
parser.add_argument("--input_setting", type=str, default='H',
                    help='the input measurement of the network: H, HM or Y')
parser.add_argument("--input_mask", type=str, default='Phi',
                    help='the input mask of the network: Phi, Phi_PhiPhiT, Mask or None')  # Phi: shift_mask   Mask: mask

# Training specifications
parser.add_argument('--batch_size', type=int, default=2, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=1, help='learning rate decay for MultiStepLR')
parser.add_argument("--epoch_sam_num", type=int, default=5000, help='the number of samples per epoch')
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--clip_grad",  type=bool, default=True)
parser.add_argument("--tune", action='store_true', help='control the max_epoch and milestones')
parser.add_argument("--loss", type=str, default='L1')
parser.add_argument("--debug",  type=bool, default=False)

# opt = parser.parse_args()
opt = parser.parse_known_args()[0]

# if opt.method == 'duf_mixs2':
#     parser = merge_duf_mixs2_opt(parser)

opt = parser.parse_known_args()[0]

template.set_template(opt)

# dataset
opt.data_path = f"{opt.train_root}/cave_1024_28/"
opt.mask_path = f"{opt.data_root}/TSA_simu_data"
opt.test_path = f"{opt.data_root}/TSA_simu_data/Truth/"

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False