import numpy as np
import torch
from .misc import AverageMeter
import SimpleITK as sitk
import os.path as osp
import medpy.metric.binary as mmb



"""
The implementation is borrowed from: https://github.com/cchen-cc/SIFA
"""
def new_validate(model, test_list, n_classes=5, gpu=True, save_dir=""):
    # function for test by volume
    dice_list = []
    assd_list = []

    model.eval()
    for idx_file, fid in enumerate(test_list):
        _npz_dict = np.load(fid)
        data = _npz_dict['arr_0']
        label = _npz_dict['arr_1']
        
        print(f'testing volume {fid}')
        if not gpu:
            model = model.cpu()
        metric_list = test_single_volume(data, label, model, n_classes, save_dir=save_dir, fid=fid)
        dice_list.append([d for d, asd in metric_list])
        assd_list.append([asd for d, asd in metric_list])

    dice_arr = 100 * np.reshape(dice_list, [-1, n_classes - 1])
    dice_mean = np.mean(dice_arr, axis=0)
    dice_std = np.std(dice_arr, axis=0)

    print('Dice:')
    print('AA :%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
    print('LAC:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
    print('LVC:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
    print('Myo:%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
    print('Mean:%.1f' % np.mean(dice_mean))

    assd_arr = np.reshape(assd_list, [-1, n_classes-1])

    assd_mean = np.mean(assd_arr, axis=0)
    assd_std = np.std(assd_arr, axis=0)

    print('ASSD:')
    print('AA :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
    print('LAC:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
    print('LVC:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
    print('Myo:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
    print('Mean:%.1f' % np.mean(assd_mean))
    
    with open(osp.join(save_dir,'result.txt'),'a') as f:
        f.write('Dice:\n')
        f.write('AA :%.1f(%.1f)\n' % (dice_mean[3], dice_std[3]))
        f.write('LAC:%.1f(%.1f)\n' % (dice_mean[1], dice_std[1]))
        f.write('LVC:%.1f(%.1f)\n' % (dice_mean[2], dice_std[2]))
        f.write('Myo:%.1f(%.1f)\n' % (dice_mean[0], dice_std[0]))
        f.write('Mean:%.1f\n' % np.mean(dice_mean))


        f.write('ASSD:\n')
        f.write('AA :%.1f(%.1f)\n' % (assd_mean[3], assd_std[3]))
        f.write('LAC:%.1f(%.1f)\n' % (assd_mean[1], assd_std[1]))
        f.write('LVC:%.1f(%.1f)\n' % (assd_mean[2], assd_std[2]))
        f.write('Myo:%.1f(%.1f)\n' % (assd_mean[0], assd_std[0]))
        f.write('Mean:%.1f\n' % np.mean(assd_mean))
        
    return np.mean(dice_mean)


def multi_validate(valloader, model, criterion, epoch, use_cuda, args):
    # function for test by slice
    losses = AverageMeter()
    model.eval()
    dice_val = np.zeros(args.num_class-1)
    with torch.no_grad():
        for batch_idx, batch in enumerate(valloader):
            inputs, targets = batch["img"].type(torch.FloatTensor), batch["gt"]
            # measure data loading time
            if use_cuda:
                targets = targets.long()
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            outputs = model(inputs)
            Lx = criterion(outputs, targets.long())
            losses.update(float(Lx.cpu().numpy()))

            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)

            for i in range(1, args.num_class):
                outputs_copy = outputs.cpu().numpy().copy()
                outputs_copy[outputs_copy != i] = 0

                targets_copy= targets.cpu().numpy().copy()
                targets_copy[targets_copy != i] = 0
                dice_val[i-1] += mmb.dc(outputs_copy, targets_copy)

    dice_val /= len(valloader)
    return losses.avg, dice_val.mean()

def one_hot_to_normal(label, dim=1):
    out = np.zeros((label.shape[:dim]) + label.shape[dim+1:], dtype=label.dtype)
    for i in range(label.shape[dim]):
        out += label[:, i] * i
    return out

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
         # ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def one_hot_encoder(input_tensor, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()


def one_hot_encoder_numpy(input, n_classes):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input == i * np.ones_like(input)
        tensor_list.append(temp_prob)
    output_tensor = np.concatenate(tensor_list, axis=1)
    return output_tensor.astype(input.dtype)


def test_single_volume(image, label, net, classes, patch_size=[256, 256], save_dir="", fid=""):
    # image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(min(image.shape)):
        if 'leuda' in fid and 'org' not in fid:
            # print('ourdata')
            slice = image[:, :, ind]
        else:
            slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        # slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            # print(out.unique())
            out = out.cpu().detach().numpy()
            pred = out
            # pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            if 'leuda' in fid and 'org' not in fid:
                prediction[...,ind] = pred
            else:
                prediction[ind] = pred

    # uncommand if want to calculate dice with largest connected components post-processing
    # prediction = post_process(prediction)

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def post_process(prediction, n_class=5):
    # connected components post-processing
    print(prediction.shape)
    prediction = one_hot_encoder_numpy(np.expand_dims(prediction, axis=1).astype(np.int32), n_class)

    for i in range(1, n_class):
        # t = nd.binary_fill_holes(t)
        prediction[:, i] = get_largest_component_sitk(prediction[:, i])
    return one_hot_to_normal(prediction, dim=1)


def get_largest_component_sitk(prediction):
    segmentation = sitk.GetImageFromArray(prediction)

    cc = sitk.ConnectedComponent(segmentation, True)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, segmentation)

    largestCClabel = 0
    largestCCsize = 0

    for l in stats.GetLabels():
        #print("Label: {0} -> Mean: {1} Size: {2}".format(l, stats.GetMean(l), int(stats.GetPhysicalSize(l))))
        if int(stats.GetPhysicalSize(l)) >= largestCCsize:
            largestCCsize = int(stats.GetPhysicalSize(l))
            largestCClabel = l

    largestCC = cc == largestCClabel  # get the largest component
    return sitk.GetArrayFromImage(largestCC).astype(np.int32)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = mmb.dc(pred, gt)
        assd = mmb.assd(pred, gt)
        return dice, assd
    else:
        return 0, 0


