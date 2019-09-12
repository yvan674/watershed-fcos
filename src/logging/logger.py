"""Logger.

Logger implementation. This will log the results of training to a file.

Features:
    - Prints out anything logged using the python Logger implementation.
    - This log is also saved.
    - Printed logs will be prepended by date/time
    - Logs a configuration file at the start of each run with the run
        configuration.
    - Sends out training curves to tensorboard.
    - Sends out one image per iteration to tensorboard along with the network
        output.
    - Sends out any printed logger outputs to tensorboard as well.
    - Tensorboard outputs are separated to dirs by run name.
"""
import logging
from os import mkdir
from os.path import isdir, join
import csv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class Logger:
    def __init__(self, work_dir, run_name, logged_objects, configuration,
                 logging_level):
        """Creates the logger object.

        Args:
            work_dir (str): Work dir being used by the trainer.
            run_name (str): Run name.
            logged_objects (list): The name of any objects that will be logged.
                e.g. ("loss_cls", "loss_bbox", "acc_cls"). Contents must be int,
                float, or str.
            configuration (dict): Training run configuration.
            logging_level (str): Logging level to use.
        """

        # Make tf dir if it doesn't exist
        if not isdir(join(work_dir, 'tf_output')):
            mkdir(join(work_dir, 'tf_output'))

        # Make the tf dir for the run
        tf_dir = join(work_dir, 'tf_output', run_name)
        mkdir(tf_dir)

        self.run_name = run_name

        # Open files and write configuration and results header
        with open(join(work_dir, run_name + '_config.txt'), 'w') as config:
            config.write("Training Parameters:\n")
            config.write("===============================================\n")
            for key in configuration.keys():
                config.write("{}: {}\n".format(key, configuration[key]))

        self.results = csv.DictWriter(open(join(work_dir,
                                                run_name + '_results.csv'),
                                           'w'), logged_objects)
        self.results.writeheader()

        # Set up python logger
        if logging_level == "DEBUG":
            logging_level = logging.WARNING
        elif logging_level == "INFO":
            logging_level = logging.INFO
        elif logging_level == "WARNING":
            logging_level = logging.WARNING
        elif logging_level == "ERROR":
            logging_level = logging.ERROR
        elif logging_level == "CRITICAL":
            logging_level = logging.CRITICAL
        else:
            raise ValueError("Logging level specified does not exist.")
        logging.basicConfig(level=logging_level,
                            file=join(work_dir, run_name + '.log'),
                            format='[%(asctime)s] :: %(levelname)s :: '
                                   '%(message)s')

        # Set up tensorboard SummaryWriter
        self.summary_writer = SummaryWriter(tf_dir)

    def log_message(self, message, logging_level):
        """Logs a message."""
        if logging_level == "DEBUG":
            logging.debug(message)
        elif logging_level == "INFO":
            logging.info(message)
        elif logging_level == "WARNING":
            logging.warning(message)
        elif logging_level == "ERROR":
            logging.error(message)
        elif logging_level == "CRITICAL":
            logging.critical(message)
        else:
            raise ValueError("Logging level specified does not exist.")

        timestamp = datetime.now().strftime('[%H:%M:%S]')
        self.summary_writer.add_text(self.run_name,
                                     '{} {} :: {}'.format(timestamp,
                                                          logging_level,
                                                          message))

    def log_results(self, objects):
        """Logs results.
        Args:
            objects (dict): Objects to be written.
        """
        self.results.writerow(objects)
        for key in objects.keys():
            self.summary_writer.add_scalar("{}/{}".format(self.run_name, key),
                                           objects[key])

    def log_images(self, images):
        """Logs images from training to tensorboard.

        Args:
            images (dict): Dict with the key being the image type and the value
                being the image itself.
        """
        for key in images.keys():
            self.summary_writer.add_image("{}/{}".format(self.run_name, key),
                                          images[key])

    def close(self):
        """Closes any files still open."""
