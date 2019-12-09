import argparse
import collections
import torch
import numpy as np
import recon_data_generator
import recon_losses_R
import recon_metrics_R
import reconmodels as module_arch
from parse_config import ConfigParser
from recon_trainer import Trainer


# TODO
#     for _,i,_ in frame_train_loader:
#        print(i)
#     def get_samples(sample_name):
#         audio = np.array(list(clip.audio.set_fps(16000).iter_frames()))
#         audio = audio.mean(1).astype(np.float32)
#         count = 0
#         sample_tuple = []
#         for frame in clip.iter_frames():
#             frame_audio = audio[count*16000//30: (count+1)*16000//30]
#             pattern = r'.+?-.+?-(.+?)-(.+?)-'
#             label = re.match(pattern, file_name).groups()
#             sample_tuple.append({'frame':frame, 'audio':frame_audio, 'emotion_class':label[0], 'emotion_intensity':label[1]})
#             count += 1
#         return sample_tuple


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('recon_data_generator', recon_data_generator)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = getattr(recon_losses_R, config['loss'])
    metrics = [getattr(recon_metrics_R, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    # main(config)
