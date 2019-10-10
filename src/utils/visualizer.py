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
from utils.colormapping import map_color_values, map_alpha_values,\
    map_bool_values
from utils.constants import *
from bboxes import draw_boxes
from bboxes.bbox import BoundingBox
from utils.sheet import process_sheet

from PIL import ImageDraw

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
            # MultiRandomFlip(0.5),
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

        self.threshold = THRESHOLD

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

                # Rearrange network out to tensor in the shape:
                # [batch, head, value, h, w]
                out_cls = self.reshape_out(out[0])
                out_bbox = self.reshape_out(out[1])
                out_scores = self.reshape_out(out[2])

                processed_images = self.process_images(image, out_bbox, out_cls,
                                                       out_scores, target)

                start_time = time()
                sheets = process_sheet(processed_images)
                self.logger.log_message("processed sheets. t={:.4f}"
                                        .format(time() - start_time))


                start_time = time()
                for sheet in sheets:
                    self.logger.log_image(sheet, 'sheet', step=data[0])
                self.logger.log_message("sent sheets to tensorboard. t={:.4f}"
                                        .format(time() - start_time))

                try:
                    val = input('Change threshold? If no, simply press enter.\n')
                    if val:
                        self.threshold = float(val)
                    else:
                        print("Not changing value.")
                except ValueError:
                    print("Not changing value.")


    def reshape_out(self, tensor_list):
        """Reshapes network to a list representing batches.

        Args:
            tensor_list (list): List of tensors given by each of the network
                outputs

        Returns:
            list: List containing a list of tensors with the shape
                out[batch][head][torch.Tensor.shape=[value, h, w]]
        """
        out_list = []
        num_heads = len(tensor_list)
        for batch in range(self.batch_size):
            out_list.append([])

        for head in range(num_heads):
            for b in range(self.batch_size):
                out_list[b].append(tensor_list[head][b])

        return out_list


    def process_images(self, raw_image, out_bbox, out_cls, out_scores, target):
        """Processes images and returns drawn version of the images overlaid
        with classes, and centerness/energy

        Args:
            out_bbox (list): list of tensors with the network output of the
                bbox predictions. Each list item represents a single batch.
            out_cls (list): list of tensor with the network output of the
                class predictions. Each list item represents a single batch.
            out_scores (list): list of tensor with the network output of the
                score predictions. Each list item represents a single batch.
            target (list): List of dicts that represents the target bboxes.

        Returns:
            list: A list of dictionaries with all the elements needed to create
                the visualization sheet. Each dictionary represents one image.
        """
        out_list = []
        for b in range(self.batch_size):
            # Turn the preds into a PIL image. First unnormalize, then turn
            # to pil image.
            img = ToPILImage()(self.unnorm(raw_image[b]))
            d = {'preds': None, 'target': None, 'score': [], 'classes': []}

            # Process bboxes
            start_time = time()
            d['preds'] = self.process_preds(img, out_bbox[b],
                                            out_cls[b], out_scores[b])

            d['target'] = self.process_target(img, target)
            self.logger.log_message("processed bounding boxes t={:.4f}"
                                    .format(time() - start_time))

            # Process scores
            start_time = time()
            d['score'] = self.process_scores(img, out_scores[b])
            self.logger.log_message("processed scores t={:.4f}"
                                    .format(time() - start_time))

            # Process classes
            start_time = time()
            d['classes'] = self.process_classes(img, out_cls[b])
            self.logger.log_message("processed classes t={:.4f}"
                                    .format(time() - start_time))
            out_list.append(d)

        return out_list

    def process_classes(self, img, out_cls):
        """Processes classes and returns an overlaid image."""
        out_imgs = []

        # Iterate through the heads
        for head in range(len(out_cls)):
            # Turn the class labels into an image
            # First get the labels as an argmax
            cls_head = out_cls[head].softmax(0).argmax(0).detach().cpu().numpy()

            # Then map the values to a color
            cls_head = map_color_values(cls_head, NUM_CLASSES)
            cls_head = Image.fromarray(cls_head)
            # And resize the cls_head to the same size as the raw image
            cls_head = cls_head.resize(img.size)

            # Finally, blend the raw image with the cls_head and append to
            # the list
            blended_img = Image.blend(img, cls_head, 0.5)
            blended_img = np.array(blended_img).astype('uint8')
            out_imgs.append(np.array(cls_head).astype('uint8'))  # Segmentation
            out_imgs.append(blended_img)  # Blended image

        return out_imgs

    def process_scores(self, img, out_scores):
        """Processes scores and returns an overlaid image."""
        energy_images = []
        for head in range(len(out_scores)):
            energy_head = out_scores[head]
            if self.model.name == "WFCOS":
                n = self.model.bbox_head.max_energy
                # If it's WFCOS, get the argmax.
                energy_head_normed = energy_head.argmax(1).numpy()
                n = energy_head_normed.max() * 3
            else:
                energy_head = energy_head.detach().cpu().numpy()
                # If it's FCOS, transpose and reshape it.
                energy_head = energy_head.transpose((1, 2, 0))
                energy_head = energy_head.reshape((energy_head.shape[0],
                                                   energy_head.shape[1]))

                # Then normalize it to the interval [0, max + min]
                energy_head_normed = energy_head + abs(energy_head.min())
                n = energy_head_normed.max() * 3

            # Map the value to colors
            energy_head_normed = map_color_values(energy_head_normed, n)
            energy_head_normed = Image.fromarray(energy_head_normed)

            # resize energy_head to the same size as the raw image
            energy_head_normed = energy_head_normed.resize(img.size)

            # If the energy head size is smaller than 20 (i.e. the 2 lowest
            # resolutions), print also the values in each pixel
            if energy_head.shape[0] < 20:
                energy_head_normed = self.draw_energy_values(energy_head_normed,
                                                             energy_head)

            # Finally blend it
            image = img.copy()
            temp_img = Image.blend(image, energy_head_normed, 0.5)

            peaks = map_bool_values(np.greater(energy_head, self.threshold),
                                           np.array([0, 0, 255]))

            # And turn it into an image
            peaks = Image.fromarray(peaks, mode='RGBA')
            # Resize it to the same size as the original image.
            peaks = peaks.resize(img.size)

            # Overlay the red on top of the original image
            energy_head_normed.paste(peaks, (0, 0), peaks)
            temp_img.paste(peaks, (0, 0), peaks)

            # Finally put it into the images list
            energy_head_normed = np.array(energy_head_normed).astype('uint8')
            energy_images.append(energy_head_normed) # Just energy
            energy_images.append(temp_img) # Blended image

        return energy_images

    def process_preds(self, img, out_bbox, out_cls, out_scores):
        """Processes bboxes and returns an overlaid image."""
        bboxes = BoundingBox()
        for head in range(len(out_bbox)):
            m = torch.tensor(
                [IMAGE_SIZE[1] / out_cls[head].shape[2],
                 IMAGE_SIZE[0] / out_cls[head].shape[1]],
                dtype=torch.float,
                device=DEVICE
            )
            bboxes.append(out_bbox[head], out_cls[head], out_scores[head], m)

        return np.array(draw_boxes(img, bboxes, threshold=self.threshold))

    def process_target(self, img, targets):
        """Processes targets and returns an overlaid image."""
        bboxes = BoundingBox()
        for t in targets:
            for b in range(self.batch_size):
                batch_bboxes = [t['bbox'][0][b],
                                t['bbox'][1][b],
                                t['bbox'][2][b],
                                t['bbox'][3][b]]
                cat = t['category_id'][b]
                bboxes.append_target(batch_bboxes, cat)
        return np.array(draw_boxes(img, bboxes, threshold=1., target=True))

    def draw_energy_values(self, image, values):
        """Draws values in values onto the image.

        Assumes that the image was created using the array values.

        Returns:
              PIL.Image.Image: The drawn image.
        """
        img = image.copy()
        """Notes:
        
        img has size (w, h)
        values has shape (h, w)
        
        """

        # First get grid size
        grid_w = img.size[0] / values.shape[1]
        grid_h = img.size[1] / values.shape[0]
        grid_half_w = grid_w / 2
        grid_half_h = grid_h / 2

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                # Iterate in left to right, then top to bottom direction.
                value = "{:.2f}".format(values[i][j])
                draw = ImageDraw.Draw(img)
                text_size = draw.textsize(value, FONT)
                text_half = (text_size[0] / 2, text_size[1] / 2)
                center_w = (grid_w * j) + grid_half_w
                center_h = (grid_h * i) + grid_half_h
                draw.rectangle([center_w - text_half[0] - 3,
                                center_h - text_half[1] - 3,
                                center_w + text_half[0] + 3,
                                center_h + text_half[1] + 3],
                               fill=(0, 0, 0))
                draw.text([center_w - text_half[0],
                           center_h - text_half[1]],
                          value,
                          fill=(255, 255, 255),
                          font=FONT)
        return img
