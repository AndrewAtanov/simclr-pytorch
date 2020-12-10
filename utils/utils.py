import numpy as np
import torch
import warnings
import time
import torch.distributed as dist


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap


def agg_all_metrics(outputs):
    if len(outputs) == 0:
        return outputs
    res = {}
    keys = [k for k in outputs[0].keys() if not isinstance(outputs[0][k], dict)]
    for k in keys:
        all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs])
        if k != 'epoch':
            res[k] = np.mean(all_logs)
        else:
            res[k] = all_logs[-1]
    return res


def gather_metrics(metrics):
    for k, v in metrics.items():
        if v.dim() == 0:
            v = v[None]
        v_all = [torch.zeros_like(v) for _ in range(dist.get_world_size())]
        dist.all_gather(v_all, v)
        v_all = torch.cat(v_all)
        metrics[k] = v_all


def viz_array_grid(array, rows, cols, padding=0, channels_last=False, normalize=False, **kwargs):
    # normalization
    '''
    Args:
        array: (N_images, N_channels, H, W) or (N_images, H, W, N_channels)
        rows, cols: rows and columns of the plot. rows * cols == array.shape[0]
        padding: padding between cells of plot
        channels_last: for Tensorflow = True, for PyTorch = False
        normalize: `False`, `mean_std`, or `min_max`
    Kwargs:
        if normalize == 'mean_std':
            mean: mean of the distribution. Default 0.5
            std: std of the distribution. Default 0.5
        if normalize == 'min_max':
            min: min of the distribution. Default array.min()
            max: max if the distribution. Default array.max()
    '''
    array = tonp(array)
    if not channels_last:
        array = np.transpose(array, (0, 2, 3, 1))

    array = array.astype('float32')

    if normalize:
        if normalize == 'mean_std':
            mean = kwargs.get('mean', 0.5)
            mean = np.array(mean).reshape((1, 1, 1, -1))
            std = kwargs.get('std', 0.5)
            std = np.array(std).reshape((1, 1, 1, -1))
            array = array * std + mean
        elif normalize == 'min_max':
            min_ = kwargs.get('min', array.min())
            min_ = np.array(min_).reshape((1, 1, 1, -1))
            max_ = kwargs.get('max', array.max())
            max_ = np.array(max_).reshape((1, 1, 1, -1))
            array -= min_
            array /= max_ + 1e-9

    batch_size, H, W, channels = array.shape
    assert rows * cols == batch_size

    if channels == 1:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1)))
        array = array[:, :, :, 0]
    elif channels == 3:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1),
                          3))
    else:
        raise TypeError('number of channels is either 1 of 3')

    for i in range(rows):
        for j in range(cols):
            img = array[i * cols + j]
            start_h = i * padding + i * H
            start_w = j * padding + j * W
            canvas[start_h: start_h + H, start_w: start_w + W] = img

    canvas = np.clip(canvas, 0, 1)
    canvas *= 255.0
    canvas = canvas.astype('uint8')
    return canvas


def tonp(x):
    if isinstance(x, (np.ndarray, float, int)):
        return np.array(x)
    return x.detach().cpu().numpy()


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        res = []
        for lr in self.base_lrs:
            res.append(np.maximum(lr * np.minimum(-self.last_epoch * 1. / self.num_epochs + 1., 1.), 0.))
        return res


class LinearWarmupAndCosineAnneal(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up, T_max, last_epoch=-1):
        self.warm_up = int(warm_up * T_max)
        self.T_max = T_max - self.warm_up
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        if self.last_epoch == 0:
            return [lr / (self.warm_up + 1) for lr in self.base_lrs]
        elif self.last_epoch <= self.warm_up:
            c = (self.last_epoch + 1) / self.last_epoch
            return [group['lr'] * c for group in self.optimizer.param_groups]
        else:
            # ref: https://github.com/pytorch/pytorch/blob/2de4f245c6b1e1c294a8b2a9d7f916d43380af4b/torch/optim/lr_scheduler.py#L493
            le = self.last_epoch - self.warm_up
            return [(1 + np.cos(np.pi * le / self.T_max)) /
                    (1 + np.cos(np.pi * (le - 1) / self.T_max)) *
                    group['lr']
                    for group in self.optimizer.param_groups]


class BaseLR(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
