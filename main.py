from __future__ import print_function
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import models.unet as models
from utils import losses, ramps
import glob
from utils import mkdir_p
from tensorboardX import SummaryWriter
from utils.utils import multi_validate, new_validate 
from dataset.dataset import WHS_dataset
import time
import logging
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

parser = argparse.ArgumentParser(description='PyTorch MT-UDA Training')

# Optimization options
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-z', '--zero_start_epochs', default=0, metavar="EPOCH", type=int)
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# Miscs
parser.add_argument('--seed', type=int, default=42, help='manual seed')

# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Method options
parser.add_argument('--val-iteration', type=int, default=1024,
                    help='Number of labeled data')

parser.add_argument('--source-path', default='',
                    help='input source path')
parser.add_argument('--target-path', default='',
                    help='input target path')
parser.add_argument('--val-path', default='',
                    help='validation path')
parser.add_argument('--val-source-path', default='',
                    help='validation path')
parser.add_argument('--test-path', default="", type=str)

parser.add_argument('--supervised_org_s_dir', default="", type=str, help='input original supervised source data path')
parser.add_argument('--supervised_cyc_s_dir', default="", type=str, help='input cycle supervised source data path')

parser.add_argument('--out', default='result',
                    help='Directory to output the result')
parser.add_argument('--batch-size', default=36, type=int)

parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--num-class', default=5, type=int)
parser.add_argument('--evaluate', action="store_true", help='set true for evaluation')

parser.add_argument('--tea_aug', action="store_true")
parser.add_argument('--sup_aug', action="store_true")

# lr
parser.add_argument("--lr_mode", default="cosine", type=str, help='cosine/poly/constant')
parser.add_argument("--lr", default=0.03, type=float)
parser.add_argument("--warmup_epochs", default=0, type=int, help='epoch number for warm up')
parser.add_argument("--change-optim", action="store_true", 
                    help='set true if want to change optimizer from Adam to SGD during training')

parser.add_argument("--optimizer", default='Adam', type=str, help='Adam/SGD')
parser.add_argument('--iter-pretrain', type=int, default=1000, help='pretraining iteration for student network')
#
parser.add_argument('--consistency_type', type=str, default="mel")
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=100, help='consistency_rampup')
parser.add_argument('--kd_type', type=str, default="mse")
parser.add_argument('--kd', type=float, default=0.1, help='kd')
parser.add_argument('--kd_rampup', type=float, default=100, help='kd_rampup')

parser.add_argument('--wo_source', action="store_true", help='without source teacher branch')
parser.add_argument('--wo_target', action="store_true", help='without target teacher branch')


parser.add_argument('--initial-lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('-Tmax', '--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training)')
parser.add_argument('--eta_min', default=0., type=float)

parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    print('deterministic')
    cudnn.benchmark = False
    cudnn.deterministic = True

# Random seed
if args.seed is None:
    args.seed = 42
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

best_acc = 0.
NUM_CLASS = args.num_class
n_iteration = 0


def print_log(*s):
    print(*s)
    logging.info(' '.join([str(st) for st in s]))


def load_to_parrallel(model, state_dict):
    out_state_dict = {}
    for k, v in state_dict.items():
        if k in model.state_dict():
            out_state_dict[k] = v
        elif k[7:] in model.state_dict():
            out_state_dict[k[7:]] = v
        else:
            out_state_dict["module." + k] = v
    model.load_state_dict(out_state_dict)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    if global_step <= args.iter_pretrain:
        alpha = 0
    else:
        alpha = min(1 - 1 / (global_step - args.iter_pretrain + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def transform_batch(imgs, gts):  # imgs, gts: [num_pairs, h, w, 2]
    # data augmentation
    imgs = imgs.squeeze(1)
    gts = gts.squeeze(1)
    for i, (img, gt) in enumerate(zip(imgs, gts)):
        seq = iaa.Affine(scale=[0.9, 1, 1, 1, 1.1], rotate=[-10, 0, 0, 0, 10])
        seq_det = seq.to_deterministic()
        segmap = SegmentationMapsOnImage(gt[..., 0], shape=gt.shape)
        img, segmaps_aug = seq_det(image=img, segmentation_maps=segmap)
        gt = segmaps_aug.arr
        gt = gt.repeat(img.shape[-1], 2)
        imgs[i] = img
        gts[i] = gt
    return imgs[:, None, ...], gts[:, None, ...]


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    source_path = args.source_path  # root/mr
    source_data = source_path.split('/')[-1]
    source_like_path = osp.join(source_path, f'{source_data}_like')  # root/mr_like

    target_path = args.target_path  # root/mr
    target_data = target_path.split('/')[-1]
    target_like_path = osp.join(target_path, f'{target_data}_like')  # root/target

    # supervised batch size / unsupervised batch size
    batch_l = args.batch_size // 4
    batch_u_s = args.batch_size // 8
    batch_u_t = args.batch_size // 16

    if not args.evaluate:
        timestamp = datetime.now().strftime("%Y.%m.%d-%H%M%S")

        writer = SummaryWriter(os.path.join("runs/" + str(args.out.split("/")[-1]), timestamp))
        writer.add_text('Text', str(args))

        log_dir = os.path.join("snapshots/" + str(args.out.split("/")[-1]))
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, timestamp), level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.info(str(args))

        print_log(f'==> Preparing dataset')

        print_log('======= Dataloader Summary =======')
        supervised_dataloaders = {
            "s": DataLoader(
                WHS_dataset(args.supervised_org_s_dir, supervised=True,   transforms=args.sup_aug),
                batch_size=batch_l, num_workers=4,
                shuffle=True),

            "cyc_s": DataLoader(
                WHS_dataset(args.supervised_cyc_s_dir, supervised=True,   transforms=args.sup_aug),
                batch_size=batch_l,
                num_workers=4, shuffle=True)
        }

        print_log('num labeled source slices (Xs): ', len(supervised_dataloaders['s'].dataset))
        print_log('num labeled source cycle slices (Xs-t-s): ', len(supervised_dataloaders['cyc_s'].dataset))

        sourcelike_dataloaders = {

            "s": DataLoader(WHS_dataset(osp.join(source_like_path, f'org_{source_data}')  ),
                            batch_size=batch_u_t,
                            num_workers=4),

            "t-s": DataLoader(WHS_dataset(osp.join(source_like_path, f'fake_{source_data}')  ),
                              batch_size=batch_u_t,
                              num_workers=4),

            "s-t-s": DataLoader(WHS_dataset(osp.join(source_like_path, f'cyc_{source_data}')  ),
                                batch_size=batch_u_t,
                                num_workers=4)
        }

        len_s = len(sourcelike_dataloaders['s'].dataset)
        len_t_s = len(sourcelike_dataloaders['t-s'].dataset)
        len_s_t_s = len(sourcelike_dataloaders['s-t-s'].dataset)

        if not args.wo_source:
            print_log('=== use source like data ===')
            print_log('num source slices (Xs): ', len_s)
            print_log('num source like slices (Xt-s): ', len_t_s)
            print_log('num source cycle slices (Xs-t-s): ', len_s_t_s)
        else:
            print_log('no source teacher used')

        targetlike_dataloaders = {
            "t": DataLoader(WHS_dataset(osp.join(target_like_path, f'org_{target_data}')  ),
                            batch_size=batch_u_t, num_workers=4),
            "s-t": DataLoader(WHS_dataset(osp.join(target_like_path, f'fake_{target_data}')  ),
                              batch_size=batch_u_t, num_workers=4),
            "t-s-t": DataLoader(WHS_dataset(osp.join(target_like_path, f'cyc_{target_data}')  ),
                                batch_size=batch_u_t, num_workers=4)
        }

        if not args.wo_target:
            print_log('=== use target like data ===')

            len_t = len(targetlike_dataloaders['t'].dataset)
            len_s_t = len(targetlike_dataloaders['s-t'].dataset)
            len_t_s_t = len(targetlike_dataloaders['t-s-t'].dataset)

            print_log('num target slices (Xt): ', len_t)
            print_log('num target like slices (Xs-t): ', len_s_t)
            print_log('num target cycle slices (Xt-s-t): ', len_t_s_t)
        else:
            print_log('no target teacher used')

        num_batch_per_epoch = len(supervised_dataloaders['s'].dataset) // batch_l
        source_total_iteration = len(sourcelike_dataloaders['s'])
        target_total_iteration = len(targetlike_dataloaders['t'])

        # iteration for update index mappings
        total_iteration = min(source_total_iteration, target_total_iteration)

        print_log('supervised dataloader batch size: ', batch_l)
        print_log('source teacher dataloader batch size: ', batch_u_s)
        print_log('sourcelike and targetlike dataloader batch size: ', batch_u_t)

        print_log('number batch per epoch:', num_batch_per_epoch)
        print_log('total iteration: ', args.epochs * num_batch_per_epoch)
        print_log('supervised total iteration per epoch: ', len(supervised_dataloaders['s']))
        print_log('source total iteration per epoch:', source_total_iteration)
        print_log('target total iteration per epoch:', target_total_iteration)
        print_log('total iteration per epoch:', total_iteration)

        val_loader = DataLoader(WHS_dataset(args.val_path, supervised=True  ), batch_size=1,
                                num_workers=2)
        val_source_loader = DataLoader(WHS_dataset(args.val_source_path, supervised=True  ),
                                       batch_size=1, num_workers=2)

    print_log("==> creating model")

    def create_model(ema=False):
        model = models.UNet(1, args.num_class)
        if len(args.gpu) > 1:
            model = nn.DataParallel(model)
        model = model.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    # SSL models
    source_ema_model = create_model(ema=True)
    target_ema_model = create_model(ema=True)

    model.train()

    print_log('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define criterion/losses

    # Supervised loss
    criterion = losses.Compose([losses.DiceLoss(args.num_class, softmax=True).cuda(), nn.CrossEntropyLoss().cuda()],
                               weights=[0.5, 0.5])

    # SSL loss
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    elif args.consistency_type == "mel":
        consistency_criterion = losses.mean_entropy_loss
    else:
        assert False, args.consistency_type

    if args.kd_type == 'mse':
        kd_criterion = losses.softmax_mse_loss
    elif args.kd_type == 'kl':
        kd_criterion = losses.softmax_kl_loss
    else:
        assert False, args.kd_type

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=0.0001)

    optimizer.zero_grad()
    start_epoch = 0

    # Resume
    if args.resume:
        print_log('==> Resuming from checkpoint..' + args.resume)
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.resume)
        # best_acc = checkpoint['best_acc']
        start_epoch = checkpoint["epoch"]
        print_log("epoch ", start_epoch)
        try:
            model.load_state_dict(checkpoint['state_dict'])
            if "source_ema_state_dict" in checkpoint and not args.wo_source:
                source_ema_model.load_state_dict(checkpoint['source_ema_state_dict'])
            if "target_ema_state_dict" in checkpoint and not args.wo_target:
                target_ema_model.load_state_dict(checkpoint['target_ema_state_dict'])
        except Exception:
            load_to_parrallel(model, checkpoint["state_dict"])
            if "source_ema_state_dict" in checkpoint and not args.wo_source:
                load_to_parrallel(source_ema_model, checkpoint["source_ema_state_dict"])
            if "target_ema_state_dict" in checkpoint and not args.wo_target:
                load_to_parrallel(target_ema_model, checkpoint["target_ema_state_dict"])
        for group in checkpoint['optimizer']["param_groups"]:
            group["betas"] = (0.9, 0.999)
            group["eps"] = 1e-8
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.evaluate:
        test_list = glob.glob(args.test_path + "/*")
        print("==> Student Results")
        new_validate(model, test_list, n_classes=args.num_class, gpu=True, save_dir=args.out  )
        if not args.wo_source:
            print("==> Source Teacher Results")
            new_validate(source_ema_model, test_list, n_classes=args.num_class, gpu=True, save_dir=args.out)
        if not args.wo_target:
            print("==> Target Teacher Results")
            new_validate(target_ema_model, test_list, n_classes=args.num_class, gpu=True, save_dir=args.out)
        return

    lr = args.lr

    sourcelike_iterators = {k: iter(tl) for k, tl in sourcelike_dataloaders.items()}
    targetlike_iterators = {k: iter(tl) for k, tl in targetlike_dataloaders.items()}

    # begin training
    for epoch in range(start_epoch, args.epochs):

        if args.warmup_epochs and epoch == args.warmup_epochs and args.change_optim and isinstance(optimizer,
                                                                                                   optim.Adam):
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  momentum=0.9, weight_decay=0.0001)
            print_log("changing to SGD")

        # train
        ###################################################### train_uda_ssl
        global global_step
        global n_iteration

        model.train()
        source_ema_model.train()
        target_ema_model.train()

        supervised_iterators = {k: iter(tl) for k, tl in supervised_dataloaders.items()}

        for batch_idx in range(num_batch_per_epoch):
            # manually shuffle the datasets, and keep datasets with same original modality have same index mapping.
            if n_iteration % total_iteration == 0:
                print_log('update index mapping...')
                del sourcelike_iterators, targetlike_iterators

                new_idx_mapping_source = {i: i_ for i, i_ in zip(range(len(sourcelike_dataloaders['s'].dataset)),
                                                                 np.random.permutation(
                                                                     range(len(sourcelike_dataloaders['s'].dataset))))}

                for dataloader in [sourcelike_dataloaders['s'], sourcelike_dataloaders['s-t-s'],
                                   targetlike_dataloaders['s-t']]:
                    dataloader.dataset.update_idx_mapping(new_idx_mapping_source)

                new_idx_mapping_target = {i: i_ for i, i_ in zip(range(len(targetlike_dataloaders['t'].dataset)),
                                                                 np.random.permutation(
                                                                     range(len(targetlike_dataloaders['t'].dataset))))}

                for dataloader in [targetlike_dataloaders['t'], sourcelike_dataloaders['t-s'],
                                   targetlike_dataloaders['t-s-t']]:
                    dataloader.dataset.update_idx_mapping(new_idx_mapping_target)

                sourcelike_iterators = {k: iter(tl) for k, tl in sourcelike_dataloaders.items()}
                targetlike_iterators = {k: iter(tl) for k, tl in targetlike_dataloaders.items()}

            n_iteration += 1
            begin = time.time()
            iter_num = batch_idx + epoch * num_batch_per_epoch
            # initialize mean teacher loss
            Lsup, Lkd, Lcon = 0., 0., 0.

            optimizer.zero_grad()

            batch_sup_s = supervised_iterators['s'].next()
            batch_sup_cyc = supervised_iterators['cyc_s'].next()

            batch_dic = {k: dataloader.next() for k, dataloader in sourcelike_iterators.items()}
            batch_dic.update({k: dataloader.next() for k, dataloader in targetlike_iterators.items()})

            for batch in [batch_sup_s, batch_sup_cyc]:
                Ls = supervised_training(batch, model, criterion)
                Lsup += Ls

            # ------ source teacher training -------------
            for k in ['s', 't-s']:
                Ls = source_teacher(batch_dic[k], model, source_ema_model, kd_criterion)  # do not cal adv loss
                Lkd += Ls

            # ------ target teacher training -------------
            for ks, kt in [('s', 's-t'), ('s-t-s', 's-t'), ('t-s', 't'), ('t-s', 't-s-t')]:
                Lt  = target_teacher(batch_dic[ks], batch_dic[kt], model, target_ema_model,
                                                    consistency_criterion)
                Lcon += Lt

            if epoch < args.warmup_epochs:
                lr = args.lr * (epoch + 1) / args.warmup_epochs  # gradual warmup_lr

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            else:
                if isinstance(optimizer, optim.SGD) and args.lr_mode == "cosine":
                    lr = adjust_learning_rate(optimizer, args.lr, epoch - args.warmup_epochs, batch_idx,
                                              num_batch_per_epoch)  # args.lr
                elif isinstance(optimizer, optim.SGD) and args.lr_mode == "poly":
                    lr = args.lr * (1.0 - epoch / args.epochs) ** 0.9
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

            if epoch < args.warmup_epochs:
                consistency_weight, kd_weight = 0., 0.
            else:
                consistency_weight = get_current_weight(epoch - args.zero_start_epochs, args.consistency,
                                                        args.consistency_rampup)
                kd_weight = get_current_weight(epoch - args.zero_start_epochs, args.kd, args.kd_rampup)

            if args.wo_source & args.wo_target:
                L = Lsup
                if n_iteration == args.iter_pretrain * num_batch_per_epoch + 1:
                    print_log('====== w/o source & w/o target =======')
            elif args.wo_target:
                L = Lsup + kd_weight * Lkd
                if n_iteration == args.iter_pretrain * num_batch_per_epoch + 1:
                    print_log('====== w/o target ssl =======')
            elif args.wo_source:
                L = Lsup + consistency_weight * Lcon
                if n_iteration == args.iter_pretrain * num_batch_per_epoch + 1:
                    print_log('====== w/o source ssl =======')
            else:
                L = Lsup + kd_weight * Lkd + consistency_weight * Lcon
                if n_iteration == args.iter_pretrain * num_batch_per_epoch + 1:
                    print_log('====== mtuda =======')

            L.backward()
            optimizer.step()

            # Update two teacher models respectively
            update_ema_variables(model, source_ema_model, args.ema_decay, iter_num)
            update_ema_variables(model, target_ema_model, args.ema_decay, iter_num)

            writer.add_scalar('losses/train_loss', L, iter_num)
            writer.add_scalar('losses/train_loss_seg', Lsup, iter_num)
            writer.add_scalar('losses/train_loss_kd', Lkd, iter_num)
            writer.add_scalar('losses/train_loss_cons', Lcon, iter_num)
            writer.add_scalar('losses/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('losses/kd_weight', kd_weight, iter_num)

            p = f"Epoch-{epoch} [{batch_idx}/{num_batch_per_epoch}]\tTot Loss: {L.item():.3f}\tSup Loss: {Lsup.item():.3f}\t"
            if not args.wo_source:
                p += f"KD Loss: {Lkd.item():.3f}\t"
            if not args.wo_target:
                p += f"Cons Loss: {Lcon.item():.3f}\tLR:{lr:.2e}\t"
            p += f"Time: {time.time() - begin:.3f}s"
            print_log(p)
            
        writer.add_scalar('lr/lr', lr, epoch)
        # test
        if (epoch+1) % 2 == 0:
            step = epoch

            stu_val_loss, stu_val_dice = multi_validate(val_loader, model, criterion, epoch, use_cuda, args)
            stea_val_loss, stea_val_dice = multi_validate(val_loader, source_ema_model, criterion, epoch, use_cuda,
                                                          args)
            ttea_val_loss, ttea_val_dice = multi_validate(val_loader, target_ema_model, criterion, epoch, use_cuda,
                                                          args)

            stu_s_val_loss, stu_s_val_dice = multi_validate(val_source_loader, model, criterion, epoch, use_cuda, args)
            stea_s_val_loss, stea_s_val_dice = multi_validate(val_source_loader, source_ema_model, criterion, epoch,
                                                              use_cuda, args)
            ttea_s_val_loss, ttea_s_val_dice = multi_validate(val_source_loader, target_ema_model, criterion, epoch,
                                                              use_cuda, args)

            writer.add_scalar('Val/stu_loss', stu_val_loss, step)
            writer.add_scalar('Model/stu_DI', stu_val_dice, step)
            writer.add_scalar('Val/source_tea_loss', stea_val_loss, step)
            writer.add_scalar('Model/source_tea_DI', stea_val_dice, step)
            writer.add_scalar('Val/target_tea_loss', ttea_val_loss, step)
            writer.add_scalar('Model/target_tea_DI', ttea_val_dice, step)

            writer.add_scalar('Val/stu_s_loss', stu_s_val_loss, step)
            writer.add_scalar('Model/stu_s_DI', stu_s_val_dice, step)
            writer.add_scalar('Val/source_tea_s_loss', stea_s_val_loss, step)
            writer.add_scalar('Model/source_tea_s_DI', stea_s_val_dice, step)
            writer.add_scalar('Val/target_tea_s_loss', ttea_s_val_loss, step)
            writer.add_scalar('Model/target_tea_s_DI', ttea_s_val_dice, step)
            # save model stu
            big_result = max(stu_val_dice, stea_val_dice, ttea_val_dice)
            best_acc = max(big_result, best_acc)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'source_ema_state_dict': source_ema_model.state_dict(),
                'target_ema_state_dict': target_ema_model.state_dict(),
                'stu_acc': round(stu_val_dice, 3),
                'source_tea_acc': round(stea_val_dice, 3),
                'target_tea_acc': round(ttea_val_dice, 3),
                'best_acc': round(best_acc, 3),
                'optimizer': optimizer.state_dict(),
            })

            print_log(
                f"Epoch-{epoch} stu Val Loss: {stu_val_loss:.3f}, stu Val Dice: {stu_val_dice:.3f}, source tea Val Loss: {stea_val_loss:.3f}, source tea Val Dice: {stea_val_dice:.3f} , target tea Val Loss: {ttea_val_loss:.3f}, target tea Val Dice: {ttea_val_dice:.3f}")
            print(f"Log dir: {args.out}, timestamp of this exp: {timestamp}")
            
    writer.close()


def supervised_training(batch_l, model, criterion):
    inputs, gts = batch_l["img"].type(torch.FloatTensor), batch_l["gt"]

    if args.tea_aug:
        inputs = inputs.numpy()[..., None]
        inputs, _ = transform_batch(inputs.copy(), inputs.copy())
        inputs = torch.tensor(inputs).type(torch.FloatTensor).squeeze(-1)
    if use_cuda:
        inputs, gts = inputs.cuda(), gts.cuda(non_blocking=True)

    logits = model(inputs)
    L_sup = criterion(logits, gts.long())

    return L_sup


def source_teacher(batch, model, ema_model, kd_criterion):
    inputs = batch["img"].type(torch.FloatTensor)

    if args.tea_aug:
        inputs = inputs.numpy()[..., None]
        inputs, _ = transform_batch(inputs.copy(), inputs.copy())
        inputs = torch.tensor(inputs).type(torch.FloatTensor).squeeze(-1)

    if use_cuda:
        inputs = inputs.cuda()

    noise = torch.clamp(torch.randn_like(inputs) * 0.1, -0.2, 0.2)
    inputs_noise = inputs + noise

    outputs = model(inputs)
    with torch.no_grad():
        outputs_ema = ema_model(inputs_noise)

    # unlabeled data
    consistency_dist = kd_criterion(outputs, outputs_ema)
    consistency_dist = torch.mean(consistency_dist)

    return consistency_dist


def target_teacher(batch_s, batch_t, model, ema_model, consistency_criterion):
    inputs_s = batch_s["img"].type(torch.FloatTensor)
    inputs_t = batch_t["img"].type(torch.FloatTensor)

    if args.tea_aug:
        inputs_ = torch.stack([inputs_s, inputs_t], -1)
        inputs_ = inputs_.numpy()
        inputs_, _ = transform_batch(inputs_.copy(), inputs_.copy())
        inputs_ = torch.tensor(inputs_).type(torch.FloatTensor)
        inputs_s, inputs_t = inputs_[..., 0], inputs_[..., 1]

    if use_cuda:
        inputs_s, inputs_t = inputs_s.cuda(), inputs_t.cuda()

    outputs_s = model(inputs_s)

    noise = torch.clamp(torch.randn_like(inputs_t) * 0.1, -0.2, 0.2)
    inputs_t_noise = inputs_t + noise
    with torch.no_grad():
        outputs_t = ema_model(inputs_t_noise)

    consistency_dist = consistency_criterion(outputs_s, outputs_t)
    consistency_dist = torch.mean(consistency_dist)

    return consistency_dist


def get_current_weight(epoch, param, rampup):
    return param * ramps.sigmoid_rampup(epoch, rampup)



def adjust_learning_rate(optimizer, lr, epoch, step_in_epoch, total_steps_in_epoch):
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr = args.eta_min + (lr - args.eta_min) * ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)
    else:
        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = ramps.linear_rampup(epoch, args.lr_rampup) * (lr - args.initial_lr) + args.initial_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def save_checkpoint(state, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


if __name__ == '__main__':
    main()