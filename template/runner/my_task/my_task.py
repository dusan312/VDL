"""
This file is the template for the boilerplate of train/test of a DNN for image classification

There are a lot of parameter which can be specified to modify the behaviour and they should be used 
instead of hard-coding stuff.
"""
import copy

import click
import cv2
#import numpy as np
#import torch
import logging
import sys
import os
import pickle
import torch

# Utils
import numpy as np

# DeepDIVA
import models
# Delegated
from torch.autograd import Variable
from torchvision import transforms
from template.runner.my_task import evaluate, train
from template.setup import set_up_model, set_up_dataloaders
from util.misc import checkpoint, adjust_learning_rate
from template.runner.apply_model import evaluate
from template.runner.apply_model.setup import set_up_dataloader
from template.setup import set_up_model
from template.runner.my_task.grad_cam import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation)

def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))


class MyTask:
    @staticmethod
    def single_run(writer, current_log_folder, model_name, epochs, lr, decay_lr,
                   output_channels, classify, **kwargs):
        """
        This is the main routine where train(), validate() and test() are called.

        Parameters
        ----------
        writer : Tensorboard.SummaryWriter
            Responsible for writing logs in Tensorboard compatible format.
        current_log_folder : string
            Path to where logs/checkpoints are saved
        model_name : string
            Name of the model
        epochs : int
            Number of epochs to train
        lr : float
            Value for learning rate
        kwargs : dict
            Any additional arguments.
        decay_lr : boolean
            Decay the lr flag
        validation_interval : int
            Run evaluation on validation set every N epochs
        checkpoint_all_epochs : bool
            If enabled, save checkpoint after every epoch.

        Returns
        -------
        train_value : ndarray[floats] of size (1, `epochs`)
            Accuracy values for train split
        val_value : ndarray[floats] of size (1, `epochs`+1)
            Accuracy values for validation split
        test_value : float
            Accuracy value for test split
        """
        CONFIG = {
            'resnet152': {
                'target_layer': 'layer4.2',
                'input_size': 224
            },
            'vgg19': {
                'target_layer': 'features.36',
                'input_size': 224
            },
            'vgg19_bn': {
                'target_layer': 'features.52',
                'input_size': 224
            },
            'inception_v3': {
                'target_layer': 'Mixed_7c',
                'input_size': 299
            },
            'densenet201': {
                'target_layer': 'features.denseblock4',
                'input_size': 224
            },
            'alexnet': {
                'target_layer': 'Convolutional_5',
                'input_size': 227
            },
            'vgg16': {
                'target_layer': 'features.30',
                'input_size': 224
            },
            # Add your model
        }.get(model_name)

        device = torch.device('cpu')

        # Get the selected model input size
        model_expected_input_size = models.__dict__[model_name]().expected_input_size
        logging.info('Model {} expects input size of {}'.format(model_name, model_expected_input_size))

        # Setting up the dataloaders
        data_loader, num_classes = set_up_dataloader(model_expected_input_size=model_expected_input_size,
                                                    classify=classify, **kwargs)

        # Setting up model, optimizer, criterion
        output_channels = num_classes if classify else output_channels

        model, _, _, _, _ = set_up_model(output_channels=output_channels,
                                         model_name=model_name,
                                         lr=lr,
                                         train_loader=None,
                                         **kwargs)
        # Synset words
        classes = list()
        with open('template/runner/my_task/samples/synset_words.txt') as lines:
            for line in lines:
                line = line.strip().split(' ', 1)[1]
                line = line.split(', ', 1)[0].replace(' ', '_')
                classes.append(line)

        # Image
        raw_image = cv2.imread('template/runner/my_task/uzorak/test_sl.png')[..., ::-1]
        raw_image = cv2.resize(raw_image, (CONFIG['input_size'], ) * 2)
        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])(raw_image).unsqueeze(0)

        logging.info('Apply GradCAM to img')
        # =========================================================================
        #print('Grad-CAM')
        # =========================================================================
        gcam = GradCAM(model=model)
        probs, idx = gcam.forward(image.to(device))

        for i in range(0, 2):
            gcam.backward(idx=idx[i])
            output = gcam.generate(target_layer=CONFIG['target_layer'])

            save_gradcam('results/{}_gcam_{}.png'.format(classes[idx[i]], 'arh'), output, raw_image)
            print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))

        return None, None, None

    ####################################################################################################################
    @staticmethod
    def _validate_model_input_size(model_expected_input_size, model_name):
        """
        This method verifies that the model expected input size is a tuple of 2 elements.
        This is necessary to avoid confusion with models which run on other types of data.

        Parameters
        ----------
        model_expected_input_size
            The item retrieved from the model which corresponds to the expected input size
        model_name : String
            Name of the model (logging purpose only)

        Returns
        -------
            None
        """
        if type(model_expected_input_size) is not tuple or len(model_expected_input_size) != 2:
            logging.error('Model {model_name} expected input size is not a tuple. '
                          'Received: {model_expected_input_size}'
                          .format(model_name=model_name,
                                  model_expected_input_size=model_expected_input_size))
            sys.exit(-1)

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package. 
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """

    @classmethod
    def _train(cls, train_loader, model, criterion, optimizer, writer, epoch, **kwargs):
        return train.train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)

    @classmethod
    def _validate(cls, val_loader, model, criterion, writer, epoch, **kwargs):
        return evaluate.validate(val_loader, model, criterion, writer, epoch, **kwargs)

    @classmethod
    def _test(cls, test_loader, model, criterion, writer, epoch, **kwargs):
        return evaluate.test(test_loader, model, criterion, writer, epoch, **kwargs)
