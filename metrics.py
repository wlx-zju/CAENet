import torch
import numpy as np


def init_criteria_ave(tasks):
    c_average = {}
    if 'semantic' in tasks:
        c_average['semantic'] = {'mIoU': 0,
                                 'pixel_acc': 0}
        c_average['IoU_classes'] = 0
    if 'depth' in tasks:
        c_average['depth'] = {'rel': 0,
                              'log10': 0,
                              'rms': 0,
                              'delta1': 0,
                              'delta2': 0,
                              'delta3': 0}
    return c_average


class CriteriaDict:
    def __init__(self, tasks):
        self.tasks = tasks
        self.criteria_sum = {'confusion_matrix': 0,
                             'depth_array': 0}
        self.criteria = init_criteria_ave(self.tasks)
        if 'semantic' in self.tasks:
            del self.criteria['IoU_classes']
        self.pixels = 0
        self.IoU_classes = 0

    def reset(self):
        for c_name in self.criteria_sum.keys():
            self.criteria_sum[c_name] = 0

    @torch.no_grad()
    def task_update(self, pred, gt, task):
        if task == 'semantic':
            class_num = pred.shape[1]
            _, pred = torch.max(pred, dim=1)
            self.criteria_sum['confusion_matrix'] += calc_confusion_matrix(pred, gt, class_num)
        elif task == 'depth':
            self.criteria_sum['depth_array'] += calc_depth_error(pred, gt)

    def update(self, pred, gt, first_epoch=False):
        for task in self.tasks:
            self.task_update(pred[task], gt[task], task)
        if first_epoch:
            if 'depth' in self.tasks:
                mask = (torch.sum(gt['depth'], dim=1) != 0)
                self.pixels += mask.sum().item()

    def evaluate(self):
        if 'semantic' in self.tasks:
            intersection = np.diag(self.criteria_sum['confusion_matrix'])
            union = self.criteria_sum['confusion_matrix'].sum(1)+self.criteria_sum['confusion_matrix'].sum(0)-intersection
            self.IoU_classes = intersection/np.maximum(1.0, union)
            self.criteria['semantic']['mIoU'] = self.IoU_classes.mean()
            self.criteria['semantic']['pixel_acc'] = intersection.sum() / self.criteria_sum['confusion_matrix'].sum()
        if 'depth' in self.tasks:
            self.criteria['depth']['rel'] = self.criteria_sum['depth_array'][0]/self.pixels
            self.criteria['depth']['log10'] = self.criteria_sum['depth_array'][1]/self.pixels
            self.criteria['depth']['rms'] = np.sqrt(self.criteria_sum['depth_array'][2]/self.pixels)
            self.criteria['depth']['delta1'] = self.criteria_sum['depth_array'][3]/self.pixels
            self.criteria['depth']['delta2'] = self.criteria_sum['depth_array'][4]/self.pixels
            self.criteria['depth']['delta3'] = self.criteria_sum['depth_array'][5]/self.pixels


def calc_confusion_matrix(pred, gt, class_num):
    mask = (gt != 255)  # ignore the unknown class (255)
    pred_valid = pred.masked_select(mask)
    gt_valid = gt.masked_select(mask)

    index = (gt_valid*class_num+pred_valid).int().cpu().numpy()  # the index in the confusion matrix, gt->row, pred->col
    confusion_matrix = np.bincount(index, minlength=class_num ** 2)
    confusion_matrix = confusion_matrix.reshape(class_num, class_num)

    return confusion_matrix


def pred_eliminate_zeros(pred_valid, eps=1e-8):
    mask = (pred_valid <= 0)
    if mask.sum() != 0:
        pred_valid[mask] = eps


def calc_depth_error(pred, gt):
    mask = (torch.sum(gt, dim=1) != 0).unsqueeze(1)
    pred_valid = pred.masked_select(mask)
    gt_valid = gt.masked_select(mask)
    depth_array = np.zeros(6)

    pred_eliminate_zeros(pred_valid)

    rel_sum = torch.sum(torch.abs(pred_valid-gt_valid)/gt_valid)
    log10_sum = torch.sum(torch.abs(torch.log10(pred_valid) - torch.log10(gt_valid)))
    square_sum = torch.sum((pred_valid - gt_valid) ** 2)

    gt_pred = (gt_valid/pred_valid).reshape(1, -1)
    pred_gt = (pred_valid/gt_valid).reshape(1, -1)
    compare_array = torch.cat((gt_pred, pred_gt), dim=0)
    ratio_max, _ = torch.max(compare_array, dim=0)
    delta1_sum = torch.sum(ratio_max < 1.25)
    delta2_sum = torch.sum(ratio_max < 1.25**2)
    delta3_sum = torch.sum(ratio_max < 1.25**3)

    depth_array[0] = rel_sum.item()
    depth_array[1] = log10_sum.item()
    depth_array[2] = square_sum.item()
    depth_array[3] = delta1_sum.item()
    depth_array[4] = delta2_sum.item()
    depth_array[5] = delta3_sum.item()

    return depth_array
