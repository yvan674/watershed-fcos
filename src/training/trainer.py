"""Trainer.

Contains the training loop.

Notes:
    Checkpoint files are written in the form: YYMMDD-HHMMSS-EEE where E is the
    epoch number. We also set the YYMMDD-HHMMSS to be the same for the
    entirety of a run.
"""
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from os.path import getctime, isfile, split, join
from glob import glob
from datetime import datetime

from model.wfcos import WFCOS
from model.loss.wfcos_calculate_loss import WFCOSLossCalculator

from utils.CocoDataset import CocoDataset

from logging.logger import Logger


class Trainer:
    def __init__(self, cfg):
        """The training loop is run by this class.

        Args:
            cfg (dict): Configuration file from train.py
        """
        # Set up checkpoint name prefix
        self.cp_prefix, self.cp = self.get_checkpoint(cfg['checkpoint'])
        self.run_name = datetime.strftime(datetime.now(), "%y%m%d-%H%M%S")
        self.cp_prefix += "/" + self.run_name
        self.start_epoch = self.cp['epoch']

        # Set up work dir
        self.work_dir = cfg['work_dir']

        self.model = WFCOS(cfg["backbone"], cfg["neck"], cfg["head"])
        self.load_model_checkpoint(cfg['pretrained'], cfg['resume'])
        self.model = nn.DataParallel(WFCOS)

        # Set up optimizer
        opt_cfg = cfg['optimizer']
        if opt_cfg['type'] == 'SGD':
            self.optimizer = SGD(lr=opt_cfg['lr'],
                                 weight_decay=opt_cfg['weight_decay'],
                                 momentum=opt_cfg['momentum'],
                                 params=self.model.parameters())
        elif opt_cfg['type'] == 'Adam':
            self.optimizer = Adam(params=self.model.parameters(),
                                  lr=opt_cfg['lr'],
                                  weight_decay=opt_cfg['weight_decay'],
                                  eps=opt_cfg['eps'])
        # TODO: Add lr scheduler

        # Set up losses
        self.loss_calc = LossCalculator(cfg['loss']['classifier'],
                                        cfg['loss']['bbox'],
                                        cfg['loss']['energy'])

        # Get datasets ready and turn them into dataloaders
        # TODO: Add transforms/augmentations
        # Notes: CocoDetection returns a tuple with the first item being the
        # image as a PIL.Image.Image object and the second item being a list of
        # the objects in the image. Each of these objects are represented as a
        # dict with keys: ['segmentation', 'area', 'iscrowd', 'image_id',
        # 'bbox', 'category_id', 'id']
        train_set = CocoDataset(join(cfg['data']['data_root'],
                                     cfg['data']['train']['img_prefix']),
                                join(cfg['data']['data_root'],
                                     cfg['data']['train']['ann_file']))
        val_set = CocoDataset(join(cfg['data']['data_root'],
                                   cfg['data']['val']['img_prefix']),
                              join(cfg['data']['data_root'],
                                   cfg['data']['val']['ann_file']))

        self.batch_size = cfg['data']['imgs_per_gpu'] * cfg['num_gpus']
        self.train_loader = DataLoader(train_set, self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, self.batch_size, shuffle=True)

        self.epochs = cfg['total_epochs']

        # Set up logging
        logged_objects = ['loss_cls', 'loss_bbox', 'loss_energy',
                          'acc_cls', 'acc_bbox', 'acc_energy',
                          'val_loss_cls', 'val_loss_bbox', 'val_loss_energy',
                          'val_acc_cls', 'val_acc_bbox', 'val_acc_energy']
        configuration = {
            'run_name': self.run_name,
            'work_dir': self.work_dir,
            'checkpoint_path': self.cp_prefix,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'model': self.model
        }
        self.logger = Logger(self.work_dir, self.run_name, logged_objects,
                             configuration, cfg['logging_level'])

    @staticmethod
    def get_checkpoint(fp):
        """Get checkpoint file from a file path, if it exists.

        First check if the checkpoint path is a file or not. If it is, then
        specify it as the chosen checkpoint. Otherwise, look for the most
        recent file in the directory. If the directory is empty, we will
        return None.

        The checkpoint file is expected to have the following keys:
        - epoch (int): The next epoch number to be executed.
        - state_dict (OrderedDict): The state_dict of the model.
        - optimizer (OrderedDict): The state_dict of the optimizer.

        Returns:
            (str, dict): A 2-tuple, with the first value being the dir path and
            the second value the file path of the checkpoint, if it exists.
        """
        #
        if isfile(fp):
            return split(fp[0]), torch.load(fp)
        else:
            checkpoints = glob(fp + '/*')
            if not checkpoints:
                return fp, None
            else:
                return fp, torch.load(max(checkpoints, key=getctime))

    def load_model_checkpoint(self, pretrained, resume):
        """Loads checkpoint/pretrained model as necessary.

        We assume that we only need to load the pretrained model if there does
        not exist a checkpoint yet.
        """
        if not self.cp or not resume:
            if pretrained:
                print("Loading pretrained backbone from {}".format(pretrained))
                self.model.load_backbone_pretrained(pretrained)
                self.model.init_weights(False, True, True)
            else:
                print("Initializing all model weights from a fresh start.")
                self.model.init_weights(True, True, True)
        else:
            print("Loading checkpoint from {}".format(self.cp))
            self.model.load_state_dict(self.cp['state_dict'])

    def load_optimizer_checkpoint(self):
        """Loads optimizer checkpoint, if necessary."""
        if self.cp:
            self.optimizer.load_state_dict(self.cp['optimizer'])

    def train(self):
        """Actually perform training."""
        for epoch in self.epochs:
            for data in enumerate(self.train_loader):
                image, target = data[1]

