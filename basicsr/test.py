import logging
import torch
from os import path as osp
import os

from tqdm import tqdm

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options, copy_opt_file
from basicsr.train import load_resume_state, mkdir_and_rename, init_tb_loggers, create_train_val_dataloader 

from basicsr.utils import img2tensor, tensor2img, imwrite

from basicsr.archs.femasr_arch import FeMaSRNet
from basicsr.utils.download_util import load_file_from_url

import pyiqa

pretrain_model_url = {
    'x4': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth',
    'x2': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth',
}

def test_pipeline(root_path):
    '''
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    '''


    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            os.makedirs(osp.join(opt['root_path'], 'tb_logger_archived'), exist_ok=True)
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    #result = create_train_val_dataloader(opt, logger)
    #train_loader, train_sampler, val_loaders, total_epochs, total_iters = result
    
    test_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        print(phase)
        if phase.split('_')[0] == 'val':
           test_set = build_dataset(dataset_opt)
           test_loader = build_dataloader(
               test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
           logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
           test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weight_path = load_file_from_url(pretrain_model_url['x4'])
    psnr = pyiqa.create_metric('psnr', device=device)
    #weight_path = 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth',

    # set up the model
    sr_model = FeMaSRNet(codebook_params=[[32, 1024, 512]], LQ_stage=True, scale_factor=4).to(device)
    sr_model.load_state_dict(torch.load(weight_path)['params'], strict=False)
    sr_model.eval()


    psnr_all = []
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        '''
        pbar = tqdm(total=len(test_loader), unit='image')
        for idx, data in enumerate(test_loader):
            lq = data['lq'].to(device)
            gt = data['gt'].to(device)
            #output = sr_model(lq)
            output = sr_model.test(lq)
            output_img = tensor2img(output)
            output_img = img2tensor(output_img).unsqueeze(0) / 255
            psnr1 = psnr(output_img, gt)
            psnr_all.append(psnr1)
            pbar.update(1)
        pbar.close()
        print("psnr", sum(psnr_all)/len(psnr_all))
        '''
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
