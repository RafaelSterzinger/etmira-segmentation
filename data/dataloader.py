import torch
from torch.utils.data import random_split
from data.augmentation import CustomMixUp, IdentityMix, RandomSampler
from data.dataset import EtMirADatasetTraining, EtMirADatasetValidation
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from config import UNLABELLED_TRAIN_PATH, TRAIN_PATH, VAL_PATH
from torchvision.transforms.v2 import CutMix, RandomChoice
from torch.utils.data import default_collate


class EtMirADataLoader(pl.LightningDataModule):
    def __init__(self, use_cps, transform_input, transform_gt, use_mix, use_augment, batch_size_train, batch_size_val, patch_size) -> None:
        super().__init__()
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_val
        self.num_workers = 32
        self.use_mix = use_mix
        self.use_cps = use_cps
        evaluation_dataset = EtMirADatasetValidation(
            data_path=VAL_PATH, transform_input=transform_input, transform_gt=transform_gt, patch_size=patch_size)

        # Set the seed for the random number generator
        torch.manual_seed(0)
        # Calculate the lengths of the validation and test datasets
        val_len = int(len(evaluation_dataset) * 0.5)  # 50% of the data
        test_len = len(evaluation_dataset) - val_len
        # Split the evaluation dataset
        self.val_dataset, self.test_dataset = random_split(
            evaluation_dataset, [val_len, test_len])

        self.train_dataset = EtMirADatasetTraining(
            data_path=TRAIN_PATH, transform_input=transform_input, transform_gt=transform_gt, use_augment=use_augment, patch_sizes=patch_size)
        if use_cps:
            self.train_no_label_dataset = EtMirADatasetTraining(
                data_path=UNLABELLED_TRAIN_PATH, transform_input=transform_input, transform_gt=transform_gt, use_augment=use_augment, patch_sizes=patch_size)

    def get_train_dataloader(self):
        cutmix = CutMix(num_classes=1)
        mixup = CustomMixUp(num_classes=1)
        identity = IdentityMix(num_classes=1)
        cutmix_or_mixup = RandomChoice([cutmix, mixup, identity])

        def collate_fn(batch):
            temp = default_collate(batch)
            # temp[0]==image, temp[1]==label
            # stack image and labels on the channel dim, i.e. gt map
            stack = torch.cat(temp, dim=1)
            mix = cutmix_or_mixup(
                stack, torch.zeros(stack.shape[0], dtype=int))
            # split image and labels again
            return [mix[0][:, 0:-1], mix[0][:, -1:]]
        dataset = {'labelled': DataLoader(self.train_dataset, batch_size=self.batch_size_train if not self.use_cps else self.batch_size_train//2, prefetch_factor=5, num_workers=self.num_workers if not self.use_cps else self.num_workers//2,
                                          shuffle=True, drop_last=True, collate_fn=collate_fn if self.use_mix else None, pin_memory=True)}
        if self.use_cps:
            dataset['unlabelled'] = DataLoader(self.train_no_label_dataset, batch_size=self.batch_size_train//2,
                                               num_workers=self.num_workers//2, prefetch_factor=5, drop_last=True, pin_memory=True, sampler=RandomSampler(self.train_no_label_dataset, num_samples=len(self.train_dataset)))
        return dataset

    def get_val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_val, num_workers=self.num_workers, shuffle=False)

    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, num_workers=self.num_workers, shuffle=False)
