import os
import time
import logging


def init_logger(args, duplicate=-1):
    log_dir = os.path.join(args.log_dir_common, args.log_dir_specific)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if duplicate >= 0:
        logger = logging.getLogger('test_{}'.format(duplicate))
    else:
        logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    f = logging.FileHandler(os.path.join(log_dir, 'log{}.txt'.format(str(time.time()))), mode='w+')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    f.setFormatter(formatter)
    logger.addHandler(f)

    l_str = '> Arguments:'
    show(l_str, logger)
    for key in args.__dict__.keys():
        l_str = '\t{}: {}'.format(key, args.__dict__[key])
        show(l_str, logger)

    return logger


def show(string, logger):
    print(string)
    logger.info(string)


def inform_logger(logger, epoch, cd_test, train_losses, tasks, time_list):
    epoch += 1
    l_str = '[epoch {}] elapsed time: {:1.2f}s, train_gt time: {:1.2f}s, test time: {:1.2f}s'.\
        format(epoch, time_list[2]-time_list[0], time_list[1]-time_list[0], time_list[2]-time_list[1])
    show(l_str, logger)
    if 'semantic' in tasks:
        l_str = '***** semantic_loss: {:1.4f} | {:1.4f} {:1.4f}'.\
            format(train_losses['semantic'], cd_test.criteria['semantic']['mIoU']*100, cd_test.criteria['semantic']['pixel_acc']*100)
        show(l_str, logger)
    if 'depth' in tasks:
        l_str = '***** depth_loss: {:1.4f} | {:1.6f} {:1.6f} {:1.6f} | {:1.4f} {:1.4f} {:1.4f}'.\
            format(train_losses['depth'], cd_test.criteria['depth']['rel'], cd_test.criteria['depth']['log10'], cd_test.criteria['depth']['rms'],
                   cd_test.criteria['depth']['delta1']*100, cd_test.criteria['depth']['delta2']*100, cd_test.criteria['depth']['delta3']*100)
        show(l_str, logger)


def close_logger(logger, c_average, tasks):
    l_str = '*********************** final test criteria ***********************'
    show(l_str, logger)

    if 'semantic' in tasks:
        l_str = 'semantic: {:1.6f} {:1.6f}'.format(c_average['semantic']['mIoU'], c_average['semantic']['pixel_acc'])
        show(l_str, logger)
        print(c_average['IoU_classes'])
        for i in range(len(c_average['IoU_classes'])):
            logger.info('*********** class{}: {:1.6f}'.format(i, c_average['IoU_classes'][i]))
    if 'depth' in tasks:
        l_str = 'depth: {:1.6f} {:1.6f} {:1.6f} | {:1.6f} {:1.6f} {:1.6f}'.\
            format(c_average['depth']['rel'], c_average['depth']['log10'], c_average['depth']['rms'],
                   c_average['depth']['delta1'], c_average['depth']['delta2'], c_average['depth']['delta3'])
        show(l_str, logger)

    loggers = list(logger.handlers)
    for i in loggers:
        logger.removeHandler(i)
        i.flush()
        i.close()
