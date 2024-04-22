import cv2
import numpy as np
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchmetrics.classification as metrics
# https://gist.github.com/pebbie/c2cec958c248339c8537e0b4b90322da
from dependencies.bwmorph_thin import bwmorph_thin as bwmorph


class PseudoFMeasure(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.f_measure = metrics.BinaryF1Score()

        self.add_state("fmeasure", default=torch.tensor(0.0),
                       dist_reduce_fx="mean")
        self.add_state("pfmeasure", default=torch.tensor(0.0),
                       dist_reduce_fx="mean")
        self.add_state("count", default=torch.tensor(0.0),
                       dist_reduce_fx="mean")
        self.sk = None
        self.prev = None

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if len(preds.shape) == 2:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)
        self.f_measure.update(preds, target)
        preds = torch.sigmoid(preds)
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        im = np.round(preds)
        im_gt = np.round(target)
        im = ~im.astype(bool)
        im_gt = ~im_gt.astype(bool)
        sk = None
        if self.prev:
            if self.prev == im_gt.sum():
                sk = self.sk
        if sk is None:
            self.prev = im_gt.sum()
            self.sk = np.stack([bwmorph(1-im_gt[i])
                               for i in range(im_gt.shape[0])])
            sk = self.sk

        im_sk = np.ones(im_gt.shape)
        im_sk[sk] = 0

        ptp = np.zeros(im_gt.shape)
        ptp[(im == 0) & (im_sk == 0)] = 1
        numptp = ptp.sum()

        tp = np.zeros(im_gt.shape)
        tp[(im == 0) & (im_gt == 0)] = 1
        numtp = tp.sum()

        tn = np.zeros(im_gt.shape)
        tn[(im == 1) & (im_gt == 1)] = 1
        numtn = tn.sum()

        fp = np.zeros(im_gt.shape)
        fp[(im == 0) & (im_gt == 1)] = 1
        numfp = fp.sum()

        fn = np.zeros(im_gt.shape)
        fn[(im == 1) & (im_gt == 0)] = 1
        numfn = fn.sum()

        np.seterr(invalid='raise')
        try:
            precision = np.divide(numtp, (numtp + numfp))
            recall = np.divide(numtp, (numtp + numfn))
            precall = np.divide(numptp, np.sum(1-im_sk))
            fmeasure = (2*recall*precision)/(recall+precision)
            pfmeasure = (2*precall*precision)/(precall+precision)
        except:
            fmeasure = 0
            pfmeasure = 0
        np.seterr(invalid='warn')

        self.fmeasure += fmeasure
        self.pfmeasure += pfmeasure
        self.count += 1.0

    def compute(self):
        result = self.pfmeasure/self.count
        self.reset()
        return result
