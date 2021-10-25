import numpy as np
import torchio as tio
import pytorch_lightning as pl
from pathlib import Path
from torchio import Subject
from torch.utils.data import DataLoader
from typing import Tuple, List
from abc import ABC, abstractmethod


class DatasetConfig:
    train_test_split_method: str = "random"  # random / sklearn / custom
    # TorchIO Augmentation parameter
    RandomGamma: float = 0.5
    RandomNoise: float = 0.5
    RandomMotion: float = 0.1
    RandomBiasField: float = 0.25


class MedicalImagingDataset(pl.LightningDataModule, ABC):
    """ Represents pytorch_lightning.LightningDataModule object which also acts as an Abstract class to make it
    flexible for any Medical Imaging Dataset """
    def __init__(self, dataset_dir, batch_size, train_label_csv, train_val_ratio, task=None, preprocess=None,
                 augmentation=None, num_workers=0):
        super().__init__()
        self.task = task
        self.dataset_dir = Path(dataset_dir)
        self.train_val_ratio = train_val_ratio
        self.augmentation = augmentation
        self.preprocess = preprocess
        self.num_workers = num_workers

        self.transforms = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    @abstractmethod
    def prepare_data(self, **kwargs) -> Tuple[List[Subject], List[Subject]]:
        """Creates a list of tio.Subject(s) for both, training and test sets."""
        pass

    def setup(self,
              data_split=DatasetConfig.train_test_split_method,
              splitting_func=None,
              kwargs=None,
              ):
        """ Set's up TRAINING, VALIDATION and TEST datasets according to the specified
        splitting criteria.
        :param data_split - how the data is going to be split, by default its random, but
         other options include: sklearn and custom
        :param splitting_func - pointer to the custom splitting function if applicable
        :param kwargs - dictionary of parameters for the custom splitting function. Must
         be passed together with splitting_func parameter
        """
        preprocess = self.get_preprocess_transform(self.preprocess)
        augmentation = self.get_augmentation_transform(self.augmentation)
        self.transforms = tio.Compose([preprocess, augmentation])

        subjects, test_subjects = self.prepare_data()

        train_subjects, val_subjects = self.split_data(subjects, data_split, splitting_func, kwargs)
        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transforms)
        # val and test dataset should not apply augmentation methods.
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(test_subjects, transform=self.preprocess)

    @staticmethod
    def get_preprocess_transform(preprocess):
        """ If specified, returns sequence of preprocessing functions. Otherwise,
        default sequence is returned.
        Default space to be resampled is T2w
        :param preprocess - sequence of custom preprocessing techniques
        :return sequence of custom, or default preprocessing techniques
        """
        if preprocess:
            return preprocess
        else:
            return tio.Compose([])

    @staticmethod
    def get_augmentation_transform(augment):
        """ If specified, returns sequence of online augmentation functions. Otherwise,
        default sequence is returned.
        :param augment - sequence of custom augmentation techniques
        :return sequence of custom, or default augmentation techniques
        """

        if augment:
            return augment
        else:
            return tio.Compose([])

    @staticmethod
    def create_tio_subject_single_image(subject_id, dicom_sequence_path, image_name, label=None) -> tio.Subject:
        """ Creates torchio.Subject and adds single image to it.
        :param subject_id: string which uniquely identifies subject id
        :param dicom_sequence_path: absolute path to DICOM sequence directory
        :param image_name: name of the image
        :param label: training label
        :returm instance of tio.Subject class
        """
        tio_image = tio.ScalarImage(dicom_sequence_path)
        subject = tio.Subject({
            "subject_id": subject_id,
            "label": label,
            image_name: tio_image
        })
        return subject

    @staticmethod
    def create_tio_subject_multiple_images(subject_id, dicom_sequence_paths, image_names, label=None):
        """ Creates torchio.Subject and adds sequence of images to it.
        :param subject_id: string which uniquely identifies subject id
        :param dicom_sequence_paths: absolute paths to DICOM sequence directories
        :param image_names: name of the images
        :param label: training label
        :returm instance of tio.Subject class
        """
        tio_image = tio.ScalarImage(dicom_sequence_paths[0])
        subject = tio.Subject({
            "subject_id": subject_id,
            "label": label,
            image_names[0]: tio_image
        })
        for idx in range(1, len(DatasetConfig.all_sequence)):
            tio_image = tio.ScalarImage(dicom_sequence_paths[idx])
            subject.add_image(image=tio_image,
                              image_name=image_names[idx])
        return subject

    @staticmethod
    def get_max_shape(subjects):
        """ Returns maximum shape based on list of subjects
        :param subjects - list of subjects for which we want
         to retrieve maximum shape
        :return maximum subject shape
        """
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def split_data(self, subjects, data_split, splitting_func, kwargs):
        """Create train and validation data"""
        num_subjects = len(subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects
        splits = num_train_subjects, num_val_subjects

        if data_split == "random":
            from torch.utils.data import random_split
            train_subjects, val_subjects = random_split(subjects, splits)
        elif data_split == "sklearn":
            from sklearn.model_selection import train_test_split
            train_subjects, val_subjects = train_test_split(subjects, test_size=self.train_val_ratio)
        elif data_split == "custom" and splitting_func and kwargs:
            train_subjects, val_subjects = splitting_func(**kwargs)
        else:
            raise ValueError(
                "[ERROR]: Please specify one of the following splitting criteria: random, sklearn or custom")
        return train_subjects, val_subjects


    def train_dataloader(self):
        """ Returns PyTorch DataLoader for TRAINING set.
        """
        return DataLoader(self.train_set, self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        """ Returns PyTorch DataLoader for VALIDATION set.
        """
        return DataLoader(self.val_set, self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """ Returns PyTorch DataLoader for TEST set.
        """
        return DataLoader(self.test_set, self.batch_size, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        """ Returns PyTorch DataLoader for Predict set.
        """
        return DataLoader(self.test_set, self.batch_size, num_workers=self.num_workers)

