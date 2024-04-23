import glob
import click
from eval import eval_model
from utils.gpu import set_cuda_precision
from data.dataloader import EtMirADataLoader
from pytorch_lightning.loggers import WandbLogger
from torchvision.transforms import ToTensor, Compose
from utils.utils import NORMALIZE
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from models.unet import UNet
from config import CHANNELS
from user_config import WANDB_ENTITY, WANDB_PROJECT

import os


@click.command()
@click.option('--device', default='gpu', help='accelarator to train on')
@click.option('--gpu', default=0, help='id of gpu')
@click.option('--use_cps', default=False, help='train on ground truth or predictions')
@click.option('--arch', default='unet', help='decoder architecture')
@click.option('--batch_train', default=32, help='batch size used for training')
@click.option('--batch_val', default=32, help='batch size used for validating')
@click.option('--lr', default=3e-4, help='learning rate')
@click.option('--ckpt', default=None, help='path to checkpoint')
@click.option('--enc_str', default='efficientnet-b6', help='type of encoder')
@click.option('--dropout', default=None, help='amount of dropout')
@click.option('--mix', default=False, help='use mixup and cutmix')
@click.option('--augment', default=True, help='use augmetation')
@click.option('--gamma', default=1.0, help='weight of unlabeled data')
@click.option('--patch_size', default=512, help='size of patch before resizing')
@click.option('--seed', default=69, help='random seed')
@click.option('--name', default='', help='name of the experiment')
@click.option('--debug', default=False, help='set debug mode')
def train_model(device, gpu, use_cps, arch, batch_train, batch_val, lr, ckpt, enc_str, dropout, mix, augment, gamma, patch_size, seed, name, debug):
    if ckpt is not None:
        if os.path.exists(ckpt) and ".ckpt" in ckpt:
            print("Resume training from checkpoint")
        else:
            raise FileNotFoundError(f"Could not locate checkpoint at {ckpt}")

    # based on not seeding workers, results will be slightly different to the ones reported
    pl.seed_everything(seed, workers=False)

    if device == 'gpu':
        set_cuda_precision()
    profiler = None

    dataloader = EtMirADataLoader(use_cps=use_cps, transform_input=Compose((ToTensor(), NORMALIZE)),
                                  transform_gt=ToTensor(),
                                  use_mix=mix,
                                  use_augment=augment,
                                  patch_size=patch_size,
                                  batch_size_train=batch_train,
                                  batch_size_val=batch_val)
    trainloader = dataloader.get_train_dataloader()
    valloader = dataloader.get_val_dataloader()
    testloader = dataloader.get_test_dataloader()

    if ckpt:
        print(f"Restoring states from {ckpt}")
        unet = UNet.load_from_checkpoint(ckpt, strict=False)
    else:
        unet = UNet(arch, len(CHANNELS), 1,
                    lr, dropout, enc_str, use_cps, gamma, patch_size)

    wandb = WandbLogger(entity=WANDB_ENTITY,
                        project=WANDB_PROJECT, name=None if not name else name)
    wandb.experiment.config['channels'] = CHANNELS

    early_stopping = EarlyStopping(monitor='metric/val/pf_measure',
                                   patience=10,
                                   mode='max',
                                   verbose=True, min_delta=0.001)
    checkpoint_callback1 = ModelCheckpoint(mode='max', filename='{epoch:02d}',
                                           save_last=True, monitor='metric/val/pf_measure')

    devices = [gpu]
    gradient_clip = 2
    trainer = pl.Trainer(max_epochs=-1 if not debug else 0, logger=wandb, log_every_n_steps=1, profiler=profiler,  
                         devices='auto' if device == 'cpu' else devices, accelerator=device, gradient_clip_val=gradient_clip, callbacks=[checkpoint_callback1, early_stopping])

    trainer.fit(model=unet, train_dataloaders=trainloader,
                val_dataloaders=valloader)

    if not debug:
        ckpt = glob.glob(os.path.join(
            wandb.name, wandb.version, 'checkpoints', 'epoch=*.ckpt'))
        ckpt = ckpt[0]
        unet = UNet.load_from_checkpoint(ckpt)
    else:
        assert ckpt
        unet = UNet.load_from_checkpoint(ckpt)
    trainer.test(model=unet, dataloaders=testloader)
    eval_model(["--device", gpu, "--ckpt",
                         ckpt, "--eval_mirror", "ANSA-VI-1700_R"])


if __name__ == '__main__':
    train_model()
