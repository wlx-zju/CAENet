import torch

from loss import init_losses, update_losses


def test_process(net, test_loader, loss_sum, cd_test, c_average, epoch, epochs, last_epochs):
    test_losses = init_losses(loss_sum.tasks)
    test_batch_num = len(test_loader)

    net.eval()
    with torch.no_grad():
        test_data = iter(test_loader)
        for k in range(test_batch_num):
            test_img_b, test_gt_b = test_data.next()
            test_pred_b = net(test_img_b)

            _, task_losses = loss_sum(test_pred_b, test_gt_b)
            update_losses(test_losses, task_losses, test_batch_num)
            cd_test.update(test_pred_b, test_gt_b, first_epoch=(epoch == 0))

        cd_test.evaluate()

        if epoch >= epochs - last_epochs:
            for task_key in cd_test.criteria.keys():
                for c_key in cd_test.criteria[task_key].keys():
                    c_average[task_key][c_key] += cd_test.criteria[task_key][c_key] / last_epochs
            if 'semantic' in loss_sum.tasks:
                c_average['IoU_classes'] += cd_test.IoU_classes / last_epochs

    return test_losses
