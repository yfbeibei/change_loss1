from __future__ import division

import os
import warnings
import torch
from config import return_args, args
torch.cuda.set_device(int(args.gpu_id[0]))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import get_root_logger, setup_seed
import nni
from nni.utils import merge_parameter
import time
import util.misc as utils
from utils import save_checkpoint
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # add tensoorboard
import cv2
import torch.nn.functional as F
from video_demo import show_map

if args.backbone == 'resnet50' or args.backbone == 'resnet101':
    from Networks.CDETR import build_model

warnings.filterwarnings('ignore')
'''fixed random seed '''
setup_seed(args.seed)

exp_save='/cluster/work/cvl/guosun/models/cod-counting/'

def main(args):
    if args['dataset'] == 'jhu':
        # test_file = './npydata/jhu_val.npy'
        test_file = './npydata/jhu_test.npy'
    elif args['dataset'] == 'cod':
        test_file = './npydata/cod_val2048.npy'
        # test_file = './npydata/cod_test2048.npy'

    with open(test_file, 'rb') as outfile:
        test_list = np.load(outfile).tolist()

    utils.init_distributed_mode(return_args)
    model, criterion, postprocessors = build_model(return_args)

    model = model.cuda()

    model = nn.DataParallel(model, device_ids=[int(data) for data in list(args['gpu_id']) if data!=','])
    # path = './save_file/log_file/debug/'
    path = exp_save+'/save_file/log_file/debug/'
    args['save_path'] = path
    if not os.path.exists(args['save_path']):
        os.makedirs(path)
    logger = get_root_logger(path + 'debug.log')
    writer = SummaryWriter(path)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("model params:", num_params / 1e6)
    logger.info("model params: = {:.3f}\t".format(num_params / 1e6))

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])
    if args['local_rank'] == 0:
        logger.info(args)

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            logger.info("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            logger.info("=> no checkpoint found at '{}'".format(args['pre']))

    print('best result:', args['best_pred'])
    logger.info('best result = {:.3f}'.format(args['best_pred']))
    torch.set_num_threads(args['workers'])

    if args['local_rank'] == 0:
        logger.info('best result={:.3f}\t start epoch={:.3f}'.format(args['best_pred'], args['start_epoch']))

    test_data = test_list
    if args['local_rank'] == 0:
        logger.info('start training!')

    eval_epoch = 0

    pred_mae, pred_mse, visi = validate(test_data, model, criterion, logger, args)
    # pred_mae, pred_mse, visi = validate2(test_data, model, criterion, logger, args)

    writer.add_scalar('Metrcis/MAE', pred_mae, eval_epoch)
    writer.add_scalar('Metrcis/MSE', pred_mse, eval_epoch)

    # save_result
    if args['save']:
        is_best = pred_mae < args['best_pred']
        args['best_pred'] = min(pred_mae, args['best_pred'])
        save_checkpoint({
            'arch': args['pre'],
            'state_dict': model.state_dict(),
            'best_prec1': args['best_pred'],
            'optimizer': optimizer.state_dict(),
        }, visi, is_best, args['save_path'])

    if args['local_rank'] == 0:
        logger.info(
            'mae={:.3f}\t mse={:.3f}\t best_mae={:.3f}\t'.format(
                args['epochs'],
                pred_mae, pred_mse,
                args['best_pred']))


def collate_wrapper(batch):
    targets = []
    imgs = []
    fname = []

    for item in batch:

        if return_args.train_patch:
            fname.append(item[0])

            for i in range(0, len(item[1])):
                imgs.append(item[1][i])

            for i in range(0, len(item[2])):
                targets.append(item[2][i])
        else:
            fname.append(item[0])
            imgs.append(item[1])
            targets.append(item[2])

    return fname, torch.stack(imgs, 0), targets


def validate(Pre_data, model, criterion, logger, args):
    if args['local_rank'] == 0:
        logger.info('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1,
    )

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    image_errs=[]
    gt_count_list = []

    for i, (fname, img, kpoint, targets, patch_info) in enumerate(test_loader):
        # img.shape: [1, 40, 3, 256, 256]
        # added by guolei
        
        assert len(patch_info)==7
        num_h, num_w=patch_info[0], patch_info[1]


        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        if len(kpoint.shape) == 5:
            kpoint = kpoint.squeeze(0)

        with torch.no_grad():
            img = img.cuda()
            outputs = model(img)
            if isinstance(outputs, list):
                outputs, out_dm=outputs[0], outputs[1]
                # count_dm
            elif isinstance(outputs, dict):
                outputs=outputs
            else:
                assert False, "unexpected output type"

        # import pdb; pdb.set_trace()
        out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
        prob = out_logits.sigmoid()
        prob = prob.view(1, -1, 2)
        out_logits = out_logits.view(1, -1, 2)
        # out_logits = out_logits.view(1, -1, 2)[:,:,1]
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1),
                                               kpoint.shape[0] * args['num_queries'], dim=1)
        count = 0
        gt_count = torch.sum(kpoint).item()
        for k in range(topk_values.shape[0]):
            sub_count = topk_values[k, :]
            sub_count[sub_count < args['threshold']] = 0
            sub_count[sub_count > 0] = 1
            sub_count = torch.sum(sub_count).item()
            count += sub_count

        if args['dm_count']:
            count_dm=0
            for k in range(out_dm[1].shape[0]):
                count_dm +=out_dm[1][k,:].sum().item()
        # if args['only_dm']:
        #     # count=count_dm
        #     count=(count+count_dm)/2
        if args['visual_path']:
            # import pdb; pdb.set_trace()
            pre_path=args['pre'].split('/')[-3]

            if not os.path.exists(os.path.join(args['visual_path'],pre_path)):
                os.makedirs(os.path.join(args['visual_path'],pre_path))

            if False:
                density_map=torch.sigmoid(out_dm[0])
                assert num_h*num_w==density_map.shape[0]
                h_dm,w_dm=density_map.shape[-2],density_map.shape[-1]
                density_map=density_map.squeeze(1).reshape(num_h,num_w,h_dm,w_dm).permute(0,2,1,3).reshape(num_h*h_dm,num_w*w_dm)
                density_map=density_map.cpu().numpy()
                density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
                density_map = (density_map * 255).astype(np.uint8)
                density_map = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
                
                ori_img=cv2.imread(os.path.join(os.path.dirname(Pre_data[0]),fname[0]))
                ori_img=torch.from_numpy(ori_img).permute(2,0,1)
                pd = (patch_info[-2], 0, patch_info[-1], 0)
                ori_img= F.pad(ori_img, pd, 'constant')
                ori_img=ori_img.permute(1,2,0).numpy()
                ori_img=cv2.resize(ori_img,(density_map.shape[1], density_map.shape[0]))

                density_map = np.concatenate((density_map, ori_img, 0.5*density_map+ori_img), axis=1)
                cv2.imwrite(os.path.join(args['visual_path'], pre_path, fname[0].replace('.jpg','_1.png')), density_map)
                print(gt_count, count, count_dm)

            if True:
                # save image patch
                # import pdb; pdb.set_trace()
                ori_img=cv2.imread(os.path.join(os.path.dirname(Pre_data[0]),fname[0]))
                ori_img=torch.from_numpy(ori_img).permute(2,0,1)
                pd = (patch_info[-2], 0, patch_info[-1], 0)
                ori_img= F.pad(ori_img, pd, 'constant')
                ori_img=ori_img.permute(1,2,0).numpy().copy()
                assert patch_info[2]==ori_img.shape[0] and patch_info[3]==ori_img.shape[1]
                ori_img2=ori_img.reshape(patch_info[0],256,patch_info[1],256,3).transpose(0,2,1,3,4).reshape(patch_info[0]*patch_info[1],256,256,3)
                for ii in range(ori_img2.shape[0]):
                    img_patch=ori_img2[ii]
                    cv2.imwrite("./visual_dm/visal_feat2/{}_im.png".format(ii), img_patch)
                exit()
            if False:
                # show map for CLTR
                # import pdb; pdb.set_trace()
                ori_img=cv2.imread(os.path.join(os.path.dirname(Pre_data[0]),fname[0]))
                ori_img=torch.from_numpy(ori_img).permute(2,0,1)
                pd = (patch_info[-2], 0, patch_info[-1], 0)
                ori_img= F.pad(ori_img, pd, 'constant')
                ori_img=ori_img.permute(1,2,0).numpy().copy()
                ori_img_ori=ori_img.copy()

                out_logits, out_point = outputs['pred_logits'], outputs['pred_points']
                prob = out_logits.sigmoid()
                topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), args['num_queries'], dim=1)

                topk_points = topk_indexes // out_logits.shape[2]
                out_point = torch.gather(out_point, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
                out_point = out_point * args['crop_size']
                value_points = torch.cat([topk_values.unsqueeze(2), out_point], 2)
                crop_size = args['crop_size']
                kpoint_map, density_map, frame, count2 = show_map(value_points, ori_img, ori_img.shape[1], ori_img.shape[0], crop_size, num_h, num_w, threshold=args['threshold'])
                res1 = np.hstack((ori_img_ori, kpoint_map))
                res2 = np.hstack((density_map, frame))
                res = np.vstack((res1, res2))
                cv2.putText(res, "Count:" + str(count2), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                cv2.putText(res, "GT:" + str(gt_count), (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                cv2.imwrite(os.path.join(args['visual_path'], pre_path, fname[0].replace('.jpg','_pred{:.2f}_gt{}.png'.format(count2,gt_count))), res)
                # assert count==count2, f'count {count}; count2 {count2}?'
                if count!=count2:
                    print(f'Not equal, count {count}; count2 {count2}?')
                print(count2)
            # if i>30:
            #     exit()

        # count=(count+count_dm)/2
        # count=count_dm
        mae += abs(count - gt_count)
        mse += abs(count - gt_count) * abs(count - gt_count)
        image_errs.append(count - gt_count)
        gt_count_list.append(gt_count)

        if i % 30 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))

    mae = mae / len(test_loader)
    mse = math.sqrt(mse / len(test_loader))

    image_errs=np.array(image_errs)
    gt_count_list=np.array(gt_count_list)
    ratio=np.divide(np.abs(image_errs),gt_count_list)
    ratio=ratio[~np.isinf(ratio)]
    ratio=ratio[~np.isnan(ratio)]
    nae=np.mean(ratio)

    print('mae', mae, 'mse', mse, 'nae', nae)
    return mae, mse, visi



if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    params = vars(merge_parameter(return_args, tuner_params))

    main(params)
