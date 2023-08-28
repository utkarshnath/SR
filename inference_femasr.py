import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite 
from basicsr.archs.femasr_arch import FeMaSRNet 
from basicsr.utils.download_util import load_file_from_url 

from math import log10, sqrt
#from basicsr.metrics import calculate_psnr
import numpy as np
import pyiqa

pretrain_model_url = {
    'x4': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth',
    'x2': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth',
}

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def main():
    """Inference demo for FeMaSR 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-w', '--weight', type=str, default=None, help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results-val', help='Output folder')
    parser.add_argument('-s', '--out_scale', type=int, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600, help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device) 

    psnr = pyiqa.create_metric('psnr', device=device)

    if args.weight is None:
        weight_path = load_file_from_url(pretrain_model_url[f'x{args.out_scale}'])
    else:
        weight_path = args.w
   
    # weight_path = "/scratch/vgunda8/FeMaSR/stage2/experiments/015_FeMaSR_LQ_stage/models/net_g_latest.pth"
    weight_path = "/scratch/vgunda8/FeMaSR/stage2/experiments/019_FeMaSR_LQ_stage_Hir_Codebook_with_Match_Selection_k1_attn_fixed/models/net_g_best_.pth"
    weight_path = "/scratch/vgunda8/FeMaSR/stage2/experiments/015_FeMaSR_LQ_stage/models/net_g_best_.pth"
    weight_path = "/scratch/vgunda8/FeMaSR/stage2/experiments/019_FeMaSR_LQ_stage_Hir_Codebook_with_Match_Selection_k1_attn_fixed_v2/models/net_g_best_.pth"
    # set up the model
    sr_model = FeMaSRNet(codebook_params=[[32, 1024, 512]], LQ_stage=True, scale_factor=args.out_scale).to(device)
    sr_model.load_state_dict(torch.load(weight_path)['params'], strict=False)
    sr_model.eval()
    
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
    
    #gt_paths =  sorted(glob.glob(os.path.join('/scratch/vgunda8/FeMaSR_Datasets/Validation_dataset/DIV2K_valid_patches', '*')))
    #lr_paths = sorted(glob.glob(os.path.join('/scratch/vgunda8/FeMaSR_Datasets/Validation_dataset_sf4/DIV2K_valid_patches', '*')))
   
    gt_paths = sorted(glob.glob(os.path.join('/scratch/vgunda8/sr_dataset/valid_div2k/HR/DIV2K_valid_HR', '*')))
    lr_paths = sorted(glob.glob(os.path.join('/scratch/vgunda8/sr_dataset/valid_div2k/LR/DIV2K_val_x4', '*')))

    
    psnr_all = []

    '''
    output_paths = sorted(glob.glob(os.path.join('/home/vgunda8/FeMaSR/results-val', '*')))
    pbar = tqdm(total=len(gt_paths), unit='image')
    for idx, (gt_path, out_path) in enumerate(zip( gt_paths, output_paths)):
        psnr1 = psnr(out_path, gt_path)
        psnr_all.append(psnr1)
        pbar.update(1)
    pbar.close()
    
    print("psnr", sum(psnr_all)/len(psnr_all))
    '''  

    pbar = tqdm(total=len(gt_paths), unit='image')
    for idx, (lr_path, gt_path) in enumerate(zip(lr_paths, gt_paths)):
        img_name = os.path.basename(lr_path)
        pbar.set_description(f'Test {img_name}')

        img = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)

        max_size = args.max_size ** 2 
        
        h, w = img_tensor.shape[2:]
        with torch.no_grad():
           if h * w < max_size: 
              output = sr_model.test(img_tensor)
           else:
              output = sr_model.test_tile(img_tensor)
        
        output_img = tensor2img(output)

        save_path = os.path.join(args.output, f'{img_name}')
        imwrite(output_img, save_path)

        img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        output_img = img2tensor(output_img).unsqueeze(0) / 255
        
        psnr1 = psnr(output_img, img_gt[None,:])
        #print(psnar1)
        #psnr = PSNR(img_gt * 255, output_img * 255)
        psnr_all.append(psnr1)

        #save_path = os.path.join(args.output, f'{img_name}')
        #imwrite(output_img, save_path)
        pbar.update(1)
    pbar.close()
    for idx, (lr_path, psnr1) in enumerate(zip(lr_paths, psnr_all)):
        img_name = os.path.basename(lr_path)
        print(img_name, psnr1)

    print("psnr", sum(psnr_all)/len(psnr_all))
    

if __name__ == '__main__':
    main()
