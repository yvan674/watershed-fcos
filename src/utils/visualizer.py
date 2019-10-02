"""Visualizer.

Visualizes Training Results.

"""
import torch
from torch.utils.data import DataLoader

from os.path import getctime, isfile, split, join
from glob import glob
from datetime import datetime
from time import time
from PIL import Image
import numpy as np

from model.wfcos import WFCOS
from model.fcos import FCOS
from transforms.multi_transforms import *
from transforms.unnormalize import UnNormalize

from torchvision.transforms.transforms import ToPILImage

from data.cocodataset import CocoDataset
from utils.colormapping import map_color_values, map_alpha_values
from utils.constants import *
from bboxes import draw_boxes
from bboxes.bbox import BoundingBox

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
        self.start_epoch = 0
        self.epochs = cfg['total_epochs']
        self.batch_size = cfg['data']['imgs_per_gpu'] \
                          * torch.cuda.device_count()

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

        self.logger.log_message("Initializing Visualizer with run name: {})"
                                .format(self.run_name))

        # Get datasets ready and turn them into dataloaders
        # Notes: CocoDetection returns a tuple with the first item being the
        # image as a PIL.Image.Image object and the second item being a list of
        # the objects in the image. Each of these objects are represented as a
        # dict with keys: ['segmentation', 'area', 'iscrowd', 'image_id',
        # 'bbox', 'category_id', 'id']
        self.img_norm = {'mean': [102.9801, 115.9465, 122.7717],
                         'std': [1.0, 1.0, 1.0]}
        transforms_to_do = MultiCompose([
            MultiResize(IMAGE_SIZE),
            MultiRandomFlip(0.5),
            MultiToTensor(),
            MultiNormalize(**self.img_norm),
        ])

        self.unnorm = UnNormalize(**self.img_norm)


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
        if CUDA_COUNT > 0:
            self.logger.log_message("Found {} GPU{}. Running using CUDA."
                                    .format(CUDA_COUNT,
                                            "s" if CUDA_COUNT > 1 else ""))
            self.model.to(DEVICE, non_blocking=True)
        else:
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
            self.logger.log_message("Loading checkpoint from file.")
            self.model.load_state_dict(self.cp['state_dict'])

    def load_optimizer_checkpoint(self):
        """Loads optimizer checkpoint, if necessary."""
        if self.cp:
            self.optimizer.load_state_dict(self.cp['optimizer'])

    def run(self):
        """Actually perform training."""
        self.logger.log_message("Starting Inference Visualizer")

        for epoch in range(self.epochs):
            for data in enumerate(self.test_loader):
                image, target = data[1]

                start_time = time()
                out = self.model(image.to(DEVICE, non_blocking=True))
                duration = time() - start_time

                """Notes:
                The network processes the image and returns a 3-member list as
                output. This list is the class, bbox, and centerness/energy
                returned.
                
                Each of these three members is a 5-tuple. The 5-tuple represents
                the different heads for each feature map scale.
                """

                self.logger.log_message("Inference ran t={:.4f} seconds"
                                        .format(duration))

                out_cls = out[0]
                out_bbox = out[1]
                out_energy = out[2]

                # Start processing bboxes
                start_time = time()
                bboxes = []  # List holding BoundingBox objects.
                for i in range(self.batch_size):
                    bboxes.append([])

                for h in range(len(out_cls)):
                    m = torch.tensor(
                        [IMAGE_SIZE[1] / out_cls[h].shape[3],
                         IMAGE_SIZE[0] / out_cls[h].shape[2]],
                        dtype=torch.float,
                        device=DEVICE
                    )
                    for b in range(self.batch_size):
                        bboxes[b].append(
                            BoundingBox(out_bbox[h][b],
                                        out_cls[h][b],
                                        out_energy[h][b],
                                        m)
                        )
                self.logger.log_message("processed bboxes t={:.4f}"
                                        .format(time() - start_time))

                # Send pictures to tensorboard
                start_time = time()
                for b in range(self.batch_size):
                    drawn_image = draw_boxes(image[b], bboxes[b])
                    drawn_image = torch.from_numpy(np.array(drawn_image))\
                        .permute(2, 0, 1)

                    self.logger.log_image(drawn_image, 'bboxes', step=data[0])
                self.logger.log_message("bboxes sent to tensorboard t={:.4f}"
                                        .format(time() - start_time))

                # Move to cpu once since we use the data on the cpu multiple
                # # times
                # out_cpu = []
                # for list in out:
                #     tensor_list = []
                #     for tensor in list:
                #         tensor_list.append(tensor.detach().cpu())
                #     out_cpu.append(tensor_list)
                #


                # # Start processing class and centerness images.
                # start_time = time()
                # img_cls, img_centerness = self.process_images(image, out_cpu)
                # duration = time() - start_time
                #
                # self.logger.log_message("processed class and centerness "
                #                         "t={:.4f})".format(duration))
                #
                # start_time = time()
                # if self.model.name == 'FCOS':
                #     # Use nms to get bboxes
                #     all_bboxes = []
                #     bbox_selected = bbox_select(out[2], .999)
                #
                #     for i in range(out[1][0].shape[0]):
                #         # Append n lists to bboxes, where n is the batch size.
                #         all_bboxes.append([])
                #
                #     for head in enumerate(out[1]):
                #         m = torch.tensor(
                #             [IMAGE_SIZE[1] / head[1][0][0].shape[1],
                #              IMAGE_SIZE[0] / head[1][0][0].shape[0]],
                #             dtype=torch.float,
                #             device=DEVICE
                #         )
                #
                #         for batch_value in range(head[1].shape[0]):
                #             # Perform operations to turn bboxes into the
                #             # (l, t, r, b) format for the logger and only take
                #             # the top nth percentile of bboxes.
                #             out_bboxes = resize_bboxes(
                #                 head[1][batch_value], m,
                #                 bbox_selected[head[0]][batch_value])
                #             # Append to the nth all_bboxes list, so the bboxes
                #             # are separated by image.
                #             all_bboxes[batch_value].append(
                #                 BoundingBox(head[1][batch_value],
                #                             head[0][batch_value],
                #                             head[2][batch_value],
                #                             m))
                # else:
                #     raise NotImplementedError
                #
                # for batch in range(len(all_bboxes)):
                #     all_bboxes[batch] = torch.cat(all_bboxes[batch], 1)
                #
                # self.logger.log_message("processed bboxes t={:.4f}"
                #                         .format(time() - start_time))
                #
                # start_time = time()
                # for i in range(len(all_bboxes)):
                #     # i is the current batch image number.
                #     # We send this to this with all the bboxes to tensorboard.
                #     self.logger.log_image(image[i],
                #                           'bboxes',
                #                           all_bboxes[i],
                #                           data[0])
                #
                # self.logger.log_message("Sent boxes to tensorboard t={:.4f}"
                #                         .format(time() - start_time))
                #
                # # Now send the segmentation and centerness overlays to
                # # tensorboard
                # start_time = time()
                # for x in img_cls:
                #     self.logger.log_image(x, 'classes',
                #                           step=data[0])
                # for x in img_centerness:
                #     self.logger.log_image(x, 'energy_centerness',
                #                           step=data[0])
                #
                # self.logger.log_message("sent classes and energy/centerness to "
                #                         "tensorboard t={:.4f}"
                #                         .format(time() - start_time))
                input('Next?')

    def process_images(self, raw_image, out):
        """Processes images and returns drawn version of the images overlaid
        with classes, and centerness/energy

        Args:
            out:
            target:

        Returns:
            tuple: A 2-tuple of the classes overlaid on the image and
                the centerness/energy overlaid on the image. Both are of type
                numpy.array. Each tuple element contains a list of the classes
                and centerness values according to each head.
        """
        # images is a 2-tuple. Each element of the tuple is a list with the
        # length being equal to the number of heads used in the model. Each
        # element of the list is thus the labeling of the corresponding head.
        cls_images = []
        energy_images = []
        num_heads = len(out[0])
        batch_size = len(raw_image)
        for batch_value in range(batch_size):
            # Turn the preds into a PIL image. First unnormalize, then turn
            # to pil image.
            img = ToPILImage()(self.unnorm(raw_image[batch_value]))

            # Iterate through the heads
            for head in range(num_heads):
                # Turn the class labels into an image
                # First get the labels as an argmax
                cls_head = out[0][head][batch_value].argmax(1).numpy()

                # Then map the values to a color
                cls_head = map_color_values(cls_head, NUM_CLASSES)
                cls_head = Image.fromarray(cls_head)
                # And resize the cls_head to the same size as the raw image
                cls_head = cls_head.resize(img.size)

                # Finally, blend the raw image with the cls_head and append to
                # the list
                temp_img = Image.blend(img, cls_head, 0.5)
                temp_img = np.array(temp_img).astype('uint8')
                temp_img = temp_img.transpose((2, 0, 1))
                cls_images.append(temp_img)

                # Now get the energy
                energy_head = out[2][head][batch_value]
                if self.model.name == "WFCOS":
                    n = self.model.bbox_head.max_energy
                    # If it's WFCOS, get the argmax.
                    energy_head = energy_head.argmax(1).numpy()
                else:
                    energy_head = energy_head.numpy()
                    # If it's FCOS, transpose and reshape it.
                    energy_head = energy_head.transpose((1, 2, 0))
                    energy_head = energy_head.reshape((energy_head.shape[0],
                                                       energy_head.shape[1]))

                    # Then normalize it to the interval [0, 1]
                    energy_head += abs(energy_head.min())
                    n = energy_head.max()

                # Map the value to colors
                energy_head = map_alpha_values(energy_head,
                                               np.array([255, 0, 0]), n)

                # And turn it into an image
                energy_head = Image.fromarray(energy_head, mode='RGBA')
                # Resize it to the same size as the original image.
                energy_head = energy_head.resize(img.size)

                # Overlay the red on top of the original image
                temp_img = img.copy()
                temp_img.paste(energy_head, (0, 0), energy_head)

                # Finally put it into the images list
                temp_img = np.array(temp_img).astype('uint8')
                temp_img = temp_img.transpose((2, 0, 1))
                energy_images.append(temp_img)

        return cls_images, energy_images
