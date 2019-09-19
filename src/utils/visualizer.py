"""Visualizer.

Visualizes Training Results.

"""
import torch
from torch.utils.data import DataLoader

from os.path import getctime, isfile, split, join
from glob import glob
from datetime import datetime
from time import time

from model.wfcos import WFCOS
from model.fcos import FCOS
from transforms.multi_transforms import *
from transforms.unnormalize import UnNormalize

from torchvision.transforms.transforms import ToPILImage, Compose

from data.cocodataset import CocoDataset

from logger.logger import Logger


class Visualizer:
    def __init__(self, cfg):
        """The training loop is run by this class.

        Args:
            cfg (dict): Configuration file from visualize.py
        """
        # Set up checkpoint name prefix
        self.cp_prefix, self.cp = self.get_checkpoint(cfg['checkpoint'])
        self.run_name = datetime.strftime(datetime.now(), "%y%m%d-%H%M%S")
        self.cp_prefix += "/" + self.run_name

        # Set up some variables
        if self.cp:
            self.start_epoch = self.cp['epoch']
        else:
            self.start_epoch = 0
        self.epochs = cfg['total_epochs']
        self.batch_size = cfg['data']['imgs_per_gpu'] * cfg['num_gpus']

        # Set up work dir
        self.work_dir = cfg['work_dir']

        # Set up model
        if cfg['model'] == "FCOS":
            self.model = FCOS(cfg["backbone"], cfg["neck"], cfg["head"])
        elif cfg['model'] == "WFCOS":
            self.model = WFCOS(cfg["backbone"], cfg["neck"], cfg["head"])
        else:
            raise ValueError("Chosen model is not implemented.")
        # Set up logger
        logged_objects = ['loss_cls', 'loss_bbox', 'loss_energy',
                          'acc_cls', 'acc_bbox', 'acc_energy',
                          'val_loss_cls', 'val_loss_bbox', 'val_loss_energy',
                          'val_acc_cls', 'val_acc_bbox', 'val_acc_energy']
        configuration = {
            'run_name': self.run_name,
            'work_dir': self.work_dir,
            'checkpoint_path': self.cp_prefix,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'model': self.model
        }
        self.logger = Logger(self.work_dir, self.run_name, logged_objects,
                             configuration, cfg['logging_level'])

        # Get datasets ready and turn them into dataloaders
        # Notes: CocoDetection returns a tuple with the first item being the
        # image as a PIL.Image.Image object and the second item being a list of
        # the objects in the image. Each of these objects are represented as a
        # dict with keys: ['segmentation', 'area', 'iscrowd', 'image_id',
        # 'bbox', 'category_id', 'id']
        self.img_norm = {'mean': [102.9801, 115.9465, 122.7717],
                         'std': [1.0, 1.0, 1.0]}
        transforms_to_do = MultiCompose([
            MultiResize((640, 800)),
            MultiRandomFlip(0.5),
            MultiToTensor(),
            MultiNormalize(**self.img_norm),
        ])


        test_set = CocoDataset(join(cfg['data']['data_root'], 'images',
                                    cfg['data']['test']['img_prefix']),
                               join(cfg['data']['data_root'],
                                    cfg['data']['test']['ann_file']),
                               transforms=transforms_to_do)
        self.test_loader = DataLoader(test_set, self.batch_size, shuffle=True)

        # Load checkpoints
        start_time = time()
        self.load_model_checkpoint(cfg['backbone']['pretrained'], cfg['resume'])
        self.logger.log_message("Loaded model and optimizer weights. ({} ms)"
                                .format(int((time() - start_time) * 1000)))

        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            cuda_count = torch.cuda.device_count()
            self.logger.log_message("Found {} GPU{}. Running using CUDA."
                                    .format(cuda_count,
                                            "s" if cuda_count > 1 else ""))
            self.model.to(self.device, non_blocking=True)
        else:
            self.device = torch.device("cpu")
            self.logger.log_message("CUDA compatible GPU not found. Running on "
                                    "CPU.", "WARNING")

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
                self.logger.log_message("Loading pretrained backbone from {}"
                                        .format(pretrained))
                self.model.load_backbone_pretrained(pretrained)
                self.model.init_weights(False, True, True)
            else:
                self.logger.log_message("Not resuming from a previous"
                                        "checkpoint. Initializing weights.",
                                        "INFO")
                self.model.init_weights(True, True, True)
        else:
            self.logger.log_message("Loading checkpoint from {}".format(self.cp))
            self.model.load_state_dict(self.cp['state_dict'])

    def load_optimizer_checkpoint(self):
        """Loads optimizer checkpoint, if necessary."""
        if self.cp:
            self.optimizer.load_state_dict(self.cp['optimizer'])

    def run(self):
        """Actually perform training."""
        self.logger.log_message("Starting Inference Visualizer")

        unnorm = UnNormalize(**self.img_norm)

        for epoch in range(self.epochs):
            for data in enumerate(self.test_loader):
                image, target = data[1]

                start_time = time()
                out = self.model(image.to(self.device, non_blocking=True))
                duration = time() - start_time
                self.logger.log_message("t={:.4f} seconds".format(duration))

                # Move to cpu once since we use the data on the cpu multiple
                # times
                out_cpu = []
                for list in out:
                    tensor_list = []
                    for tensor in list:
                        tensor_list.append(tensor.detach().cpu())
                    out_cpu.append(tensor_list)

                for i in range(len(target)):
                    for j in range(len(target[i]['image_id'])):
                        current_image = image[j]
                        self.logger.log_images(
                            {str(target[i]['image_id'][j].item()) + "_normed":
                                 current_image.numpy()})
                        unnormed = unnorm(current_image)
                        self.logger.log_images(
                            {str(target[i]['image_id'][j].item()) + '_unnormed':
                                 unnormed.numpy()})
                input('Next?')
