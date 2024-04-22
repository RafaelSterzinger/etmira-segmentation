
from torch.utils.data import Sampler
import torchvision.tv_tensors as tv_tensors
from torchvision.transforms.v2._augment import _BaseMixUpCutMix
from typing import Dict, Any, Iterator, List, Optional, Sized
import torch


def is_pure_tensor(inpt: Any) -> bool:
    return isinstance(inpt, torch.Tensor) and not isinstance(inpt, tv_tensors.TVTensor)


class IdentityMix(_BaseMixUpCutMix):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt


class CustomMixUp(_BaseMixUpCutMix):
    """[BETA] Apply MixUp to the provided batch of images and labels.

    .. v2betastatus:: MixUp transform

    Paper: `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int): number of classes in the batch. Used for one-hot-encoding.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``MixUp()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())))  # type: ignore[arg-type]

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        lam = params["lam"]
        lam = 0.4 + lam*0.2

        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=lam)
        elif isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)) or is_pure_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params["batch_size"])
            img = inpt[:, 0:-1]
            label = inpt[:, -1:]
            output_img = img.roll(1, 0).mul_(
                1.0 - lam).add_(img.mul(lam))
            output_label = torch.maximum(label.roll(1, 0), label)
            output = torch.cat([output_img, output_label],
                               dim=1)

            if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
                output = tv_tensors.wrap(output, like=inpt)

            return output
        else:
            return inpt


class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized

    def __init__(self, data_source: Sized, num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

    def __len__(self) -> int:
        return self.num_samples
