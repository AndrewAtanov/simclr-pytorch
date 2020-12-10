from argparse import Namespace, ArgumentParser

import os
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from utils import datautils
import models
from utils import utils
import numpy as np
import PIL
from tqdm import tqdm
import sklearn
from utils.lars_optimizer import LARS
import scipy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import copy

class BaseSSL(nn.Module):
    """
    Inspired by the PYTORCH LIGHTNING https://pytorch-lightning.readthedocs.io/en/latest/
    Similar but lighter and customized version.
    """
    DATA_ROOT = os.environ.get('DATA_ROOT', os.path.dirname(os.path.abspath(__file__)) + '/data')
    IMAGENET_PATH = os.environ.get('IMAGENET_PATH', '/home/aashukha/imagenet/raw-data/')

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        if hparams.data == 'imagenet':
            print(f"IMAGENET_PATH = {self.IMAGENET_PATH}")

    def get_ckpt(self):
        return {
            'state_dict': self.state_dict(),
            'hparams': self.hparams,
        }

    @classmethod
    def load(cls, ckpt, device=None):
        parser = ArgumentParser()
        cls.add_model_hparams(parser)
        hparams = parser.parse_args([], namespace=ckpt['hparams'])

        res = cls(hparams, device=device)
        res.load_state_dict(ckpt['state_dict'])
        return res

    @classmethod
    def default(cls, device=None, **kwargs):
        parser = ArgumentParser()
        cls.add_model_hparams(parser)
        hparams = parser.parse_args([], namespace=Namespace(**kwargs))
        res = cls(hparams, device=device)
        return res

    def forward(self, x):
        pass

    def transforms(self):
        pass

    def samplers(self):
        return None, None

    def prepare_data(self):
        train_transform, test_transform = self.transforms()
        # print('The following train transform is used:\n', train_transform)
        # print('The following test transform is used:\n', test_transform)
        if self.hparams.data == 'cifar':
            self.trainset = datasets.CIFAR10(root=self.DATA_ROOT, train=True, download=True, transform=train_transform)
            self.testset = datasets.CIFAR10(root=self.DATA_ROOT, train=False, download=True, transform=test_transform)
        elif self.hparams.data == 'imagenet':
            traindir = os.path.join(self.IMAGENET_PATH, 'train')
            valdir = os.path.join(self.IMAGENET_PATH, 'val')
            self.trainset = datasets.ImageFolder(traindir, transform=train_transform)
            self.testset = datasets.ImageFolder(valdir, transform=test_transform)
        else:
            raise NotImplementedError

    def dataloaders(self, iters=None):
        train_batch_sampler, test_batch_sampler = self.samplers()
        if iters is not None:
            train_batch_sampler = datautils.ContinousSampler(
                train_batch_sampler,
                iters
            )

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_sampler=train_batch_sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            self.testset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_sampler=test_batch_sampler,
        )

        return train_loader, test_loader

    @staticmethod
    def add_parent_hparams(add_model_hparams):
        def foo(cls, parser):
            for base in cls.__bases__:
                base.add_model_hparams(parser)
            add_model_hparams(cls, parser)
        return foo

    @classmethod
    def add_model_hparams(cls, parser):
        parser.add_argument('--data', help='Dataset to use', default='cifar')
        parser.add_argument('--arch', default='ResNet50', help='Encoder architecture')
        parser.add_argument('--batch_size', default=256, type=int, help='The number of unique images in the batch')
        parser.add_argument('--aug', default=True, type=bool, help='Applies random augmentations if True')


class SimCLR(BaseSSL):
    @classmethod
    @BaseSSL.add_parent_hparams
    def add_model_hparams(cls, parser):
        # loss params
        parser.add_argument('--temperature', default=0.1, type=float, help='Temperature in the NTXent loss')
        # data params
        parser.add_argument('--multiplier', default=2, type=int)
        parser.add_argument('--color_dist_s', default=1., type=float, help='Color distortion strength')
        parser.add_argument('--scale_lower', default=0.08, type=float, help='The minimum scale factor for RandomResizedCrop')
        # ddp
        parser.add_argument('--sync_bn', default=True, type=bool,
            help='Syncronises BatchNorm layers between all processes if True'
        )

    def __init__(self, hparams, device=None):
        super().__init__(hparams)

        self.hparams.dist = getattr(self.hparams, 'dist', 'dp')

        model = models.encoder.EncodeProject(hparams)
        self.reset_parameters()
        if device is not None:
            model = model.to(device)
        if self.hparams.dist == 'ddp':
            if self.hparams.sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            dist.barrier()
            if device is not None:
                model = model.to(device)
            self.model = DDP(model, [hparams.gpu], find_unused_parameters=True)
        elif self.hparams.dist == 'dp':
            self.model = nn.DataParallel(model)
        else:
            raise NotImplementedError

        self.criterion = models.losses.NTXent(
            tau=hparams.temperature,
            multiplier=hparams.multiplier,
            distributed=(hparams.dist == 'ddp'),
        )

    def reset_parameters(self):
        def conv2d_weight_truncated_normal_init(p):
            fan_in = p.shape[1]
            stddev = np.sqrt(1. / fan_in) / .87962566103423978
            r = scipy.stats.truncnorm.rvs(-2, 2, loc=0, scale=1., size=p.shape)
            r = stddev * r
            with torch.no_grad():
                p.copy_(torch.FloatTensor(r))

        def linear_normal_init(p):
            with torch.no_grad():
                p.normal_(std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv2d_weight_truncated_normal_init(m.weight)
            elif isinstance(m, nn.Linear):
                linear_normal_init(m.weight)

    def step(self, batch):
        x, _ = batch
        z = self.model(x)
        loss, acc = self.criterion(z)
        return {
            'loss': loss,
            'contrast_acc': acc,
        }

    def encode(self, x):
        return self.model(x, out='h')

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_step(self, batch, it=None):
        logs = self.step(batch)

        if self.hparams.dist == 'ddp':
            self.trainsampler.set_epoch(it)
        if it is not None:
            logs['epoch'] = it / len(self.batch_trainsampler)

        return logs

    def test_step(self, batch):
        return self.step(batch)

    def samplers(self):
        if self.hparams.dist == 'ddp':
            # trainsampler = torch.utils.data.distributed.DistributedSampler(self.trainset, num_replicas=1, rank=0)
            trainsampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
            print(f'Process {dist.get_rank()}: {len(trainsampler)} training samples per epoch')
            testsampler = torch.utils.data.distributed.DistributedSampler(self.testset)
            print(f'Process {dist.get_rank()}: {len(testsampler)} test samples')
        else:
            trainsampler = torch.utils.data.sampler.RandomSampler(self.trainset)
            testsampler = torch.utils.data.sampler.RandomSampler(self.testset)

        batch_sampler = datautils.MultiplyBatchSampler
        # batch_sampler.MULTILPLIER = self.hparams.multiplier if self.hparams.dist == 'dp' else 1
        batch_sampler.MULTILPLIER = self.hparams.multiplier

        # need for DDP to sync samplers between processes
        self.trainsampler = trainsampler
        self.batch_trainsampler = batch_sampler(trainsampler, self.hparams.batch_size, drop_last=True)

        return (
            self.batch_trainsampler,
            batch_sampler(testsampler, self.hparams.batch_size, drop_last=True)
        )

    def transforms(self):
        if self.hparams.data == 'cifar':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    32,
                    scale=(self.hparams.scale_lower, 1.0),
                    interpolation=PIL.Image.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                datautils.get_color_distortion(s=self.hparams.color_dist_s),
                transforms.ToTensor(),
                datautils.Clip(),
            ])
            test_transform = train_transform

        elif self.hparams.data == 'imagenet':
            from utils.datautils import GaussianBlur

            im_size = 224
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    im_size,
                    scale=(self.hparams.scale_lower, 1.0),
                    interpolation=PIL.Image.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(0.5),
                datautils.get_color_distortion(s=self.hparams.color_dist_s),
                transforms.ToTensor(),
                GaussianBlur(im_size // 10, 0.5),
                datautils.Clip(),
            ])
            test_transform = train_transform
        return train_transform, test_transform

    def get_ckpt(self):
        return {
            'state_dict': self.model.module.state_dict(),
            'hparams': self.hparams,
        }

    def load_state_dict(self, state):
        k = next(iter(state.keys()))
        if k.startswith('model.module'):
            super().load_state_dict(state)
        else:
            self.model.module.load_state_dict(state)


class SSLEval(BaseSSL):
    @classmethod
    @BaseSSL.add_parent_hparams
    def add_model_hparams(cls, parser):
        parser.add_argument('--test_bs', default=256, type=int)
        parser.add_argument('--encoder_ckpt', default='', help='Path to the encoder checkpoint')
        parser.add_argument('--precompute_emb_bs', default=-1, type=int,
            help='If it\'s not equal to -1 embeddings are precomputed and fixed before training with batch size equal to this.'
        )
        parser.add_argument('--finetune', default=False, type=bool, help='Finetunes the encoder if True')
        parser.add_argument('--augmentation', default='RandomResizedCrop', help='')
        parser.add_argument('--scale_lower', default=0.08, type=float, help='The minimum scale factor for RandomResizedCrop')

    def __init__(self, hparams, device=None):
        super().__init__(hparams)

        self.hparams.dist = getattr(self.hparams, 'dist', 'dp')

        if hparams.encoder_ckpt != '':
            ckpt = torch.load(hparams.encoder_ckpt, map_location=device)
            if getattr(ckpt['hparams'], 'dist', 'dp') == 'ddp':
                ckpt['hparams'].dist = 'dp'
            if self.hparams.dist == 'ddp':
                ckpt['hparams'].dist = 'gpu:%d' % hparams.gpu

            self.encoder = models.REGISTERED_MODELS[ckpt['hparams'].problem].load(ckpt, device=device)
        else:
            print('===> Random encoder is used!!!')
            self.encoder = SimCLR.default(device=device)
        self.encoder.to(device)

        if not hparams.finetune:
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif hparams.dist == 'ddp':
            raise NotImplementedError

        self.encoder.eval()
        if hparams.data == 'cifar':
            hdim = self.encode(torch.ones(10, 3, 32, 32).to(device)).shape[1]
            n_classes = 10
        elif hparams.data == 'imagenet':
            hdim = self.encode(torch.ones(10, 3, 224, 224).to(device)).shape[1]
            n_classes = 1000

        if hparams.arch == 'linear':
            model = nn.Linear(hdim, n_classes).to(device)
            model.weight.data.zero_()
            model.bias.data.zero_()
            self.model = model
        else:
            raise NotImplementedError

        if hparams.dist == 'ddp':
            self.model = DDP(model, [hparams.gpu])

    def encode(self, x):
        return self.encoder.model(x, out='h')

    def step(self, batch):
        if self.hparams.problem == 'eval' and self.hparams.data == 'imagenet':
            batch[0] = batch[0] / 255.
        h, y = batch
        if self.hparams.precompute_emb_bs == -1:
            h = self.encode(h)
        p = self.model(h)
        loss = F.cross_entropy(p, y)
        acc = (p.argmax(1) == y).float()
        return {
            'loss': loss,
            'acc': acc,
        }

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train_step(self, batch, it=None):
        logs = self.step(batch)
        if it is not None:
            iters_per_epoch = len(self.trainset) / self.hparams.batch_size
            iters_per_epoch = max(1, int(np.around(iters_per_epoch)))
            logs['epoch'] = it / iters_per_epoch
        if self.hparams.dist == 'ddp' and self.hparams.precompute_emb_bs == -1:
            self.object_trainsampler.set_epoch(it)

        return logs

    def test_step(self, batch):
        logs = self.step(batch)
        if self.hparams.dist == 'ddp':
            utils.gather_metrics(logs)
        return logs

    def prepare_data(self):
        super().prepare_data()

        def create_emb_dataset(dataset):
            embs, labels = [], []
            loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=self.hparams.workers,
                pin_memory=True,
                batch_size=self.hparams.precompute_emb_bs,
                shuffle=False,
            )
            for x, y in tqdm(loader):
                if self.hparams.data == 'imagenet':
                    x = x.to(torch.device('cuda'))
                    x = x / 255.
                e = self.encode(x)
                embs.append(utils.tonp(e))
                labels.append(utils.tonp(y))
            embs, labels = np.concatenate(embs), np.concatenate(labels)
            dataset = torch.utils.data.TensorDataset(torch.FloatTensor(embs), torch.LongTensor(labels))
            return dataset

        if self.hparams.precompute_emb_bs != -1:
            print('===> Precompute embeddings:')
            assert not self.hparams.aug
            with torch.no_grad():
                self.encoder.eval()
                self.testset = create_emb_dataset(self.testset)
                self.trainset = create_emb_dataset(self.trainset)
        
        print(f'Train size: {len(self.trainset)}')
        print(f'Test size: {len(self.testset)}')

    def dataloaders(self, iters=None):
        if self.hparams.dist == 'ddp' and self.hparams.precompute_emb_bs == -1:
            trainsampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
            testsampler = torch.utils.data.distributed.DistributedSampler(self.testset, shuffle=False)
        else:
            trainsampler = torch.utils.data.RandomSampler(self.trainset)
            testsampler = torch.utils.data.SequentialSampler(self.testset)

        self.object_trainsampler = trainsampler
        trainsampler = torch.utils.data.BatchSampler(
            self.object_trainsampler,
            batch_size=self.hparams.batch_size, drop_last=False,
        )
        if iters is not None:
            trainsampler = datautils.ContinousSampler(trainsampler, iters)

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            batch_sampler=trainsampler,
        )
        test_loader = torch.utils.data.DataLoader(
            self.testset,
            num_workers=self.hparams.workers,
            pin_memory=True,
            sampler=testsampler,
            batch_size=self.hparams.test_bs,
        )
        return train_loader, test_loader

    def transforms(self):
        if self.hparams.data == 'cifar':
            trs = []
            if 'RandomResizedCrop' in self.hparams.augmentation:
                trs.append(
                    transforms.RandomResizedCrop(
                        32,
                        scale=(self.hparams.scale_lower, 1.0),
                        interpolation=PIL.Image.BICUBIC,
                    )
                )
            if 'RandomCrop' in self.hparams.augmentation:
                trs.append(transforms.RandomCrop(32, padding=4, padding_mode='reflect'))
            if 'color_distortion' in self.hparams.augmentation:
                trs.append(datautils.get_color_distortion(self.encoder.hparams.color_dist_s))

            train_transform = transforms.Compose(trs + [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                datautils.Clip(),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif self.hparams.data == 'imagenet':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    224,
                    scale=(self.hparams.scale_lower, 1.0),
                    interpolation=PIL.Image.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                lambda x: (255*x).byte(),
            ])
            test_transform = transforms.Compose([
                datautils.CenterCropAndResize(proportion=0.875, size=224),
                transforms.ToTensor(),
                lambda x: (255 * x).byte(),
            ])
        return train_transform if self.hparams.aug else test_transform, test_transform

    def train(self, mode=True):
        if self.hparams.finetune:
            super().train(mode)
        else:
            self.model.train(mode)

    def get_ckpt(self):
        return {
            'state_dict': self.state_dict() if self.hparams.finetune else self.model.state_dict(),
            'hparams': self.hparams,
        }

    def load_state_dict(self, state):
        if self.hparams.finetune:
            super().load_state_dict(state)
        else:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(state)
            else:
                self.model.load_state_dict(state)

class SemiSupervisedEval(SSLEval):
    @classmethod
    @BaseSSL.add_parent_hparams
    def add_model_hparams(cls, parser):
        parser.add_argument('--train_size', default=-1, type=int)
        parser.add_argument('--data_split_seed', default=42, type=int)
        parser.add_argument('--n_augs_train', default=-1, type=int)
        parser.add_argument('--n_augs_test', default=-1, type=int)
        parser.add_argument('--acc_on_unlabeled', default=False, type=bool)

    def prepare_data(self):
        super(SSLEval, self).prepare_data()

        if len(self.trainset) != self.hparams.train_size:
            idxs, unlabeled_idxs = sklearn.model_selection.train_test_split(
                np.arange(len(self.trainset)),
                train_size=self.hparams.train_size,
                random_state=self.hparams.data_split_seed,
            )
            if self.hparams.data == 'cifar' or self.hparams.data == 'cifar100':
                if self.hparams.acc_on_unlabeled:
                    self.trainset_unlabeled = copy.deepcopy(self.trainset)
                    self.trainset_unlabeled.data = self.trainset.data[unlabeled_idxs]
                    self.trainset_unlabeled.targets = np.array(self.trainset.targets)[unlabeled_idxs]
                    print(f'Test size (0): {len(self.testset)}')
                    print(f'Unlabeled train size (1):  {len(self.trainset_unlabeled)}')

                self.trainset.data = self.trainset.data[idxs]
                self.trainset.targets = np.array(self.trainset.targets)[idxs]

                print('Training dataset size:', len(self.trainset))
            else:
                assert not self.hparams.acc_on_unlabeled
                if isinstance(self.trainset, torch.utils.data.TensorDataset):
                    self.trainset.tensors = [t[idxs] for t in self.trainset.tensors]
                else:
                    self.trainset.samples = [self.trainset.samples[i] for i in idxs]

                print('Training dataset size:', len(self.trainset))

        self.encoder.eval()
        with torch.no_grad():
            if self.hparams.n_augs_train != -1:
                self.trainset = EmbEnsEval.create_emb_dataset(self, self.trainset, n_augs=self.hparams.n_augs_train)
            if self.hparams.n_augs_test != -1:
                self.testset = EmbEnsEval.create_emb_dataset(self, self.testset, n_augs=self.hparams.n_augs_test)
                if self.hparams.acc_on_unlabeled:
                    self.trainset_unlabeled = EmbEnsEval.create_emb_dataset(
                        self,
                        self.trainset_unlabeled,
                        n_augs=self.hparams.n_augs_test
                    )
        if self.hparams.acc_on_unlabeled:
            self.testset = torch.utils.data.ConcatDataset([
                datautils.DummyOutputWrapper(self.testset, 0),
                datautils.DummyOutputWrapper(self.trainset_unlabeled, 1)
            ])

    def transforms(self):
        ens_train_transfom, ens_test_transform = EmbEnsEval.transforms(self)
        train_transform, test_transform = SSLEval.transforms(self)
        return (
            train_transform if self.hparams.n_augs_train == -1 else ens_train_transfom,
            test_transform if self.hparams.n_augs_test == -1 else ens_test_transform
        )

    def step(self, batch, it=None):
        if self.hparams.problem == 'eval' and self.hparams.data == 'imagenet':
            batch[0] = batch[0] / 255.
        h, y = batch
        if len(h.shape) == 4:
            h = self.encode(h)
        p = self.model(h)
        loss = F.cross_entropy(p, y)
        acc = (p.argmax(1) == y).float()
        return {
            'loss': loss,
            'acc': acc,
        }

    def test_step(self, batch):
        if not self.hparams.acc_on_unlabeled:
            return super().test_step(batch)
        # TODO: refactor
        x, y, d = batch
        logs = {}
        keys = set()
        for didx in [0, 1]:
            if torch.any(d == didx):
                t = super().test_step([x[d == didx], y[d == didx]])
                for k, v in t.items():
                    keys.add(k)
                    logs[k + f'_{didx}'] = v
        for didx in [0, 1]:
            for k in keys:
                logs[k + f'_{didx}'] = logs.get(k + f'_{didx}', torch.tensor([]))
        return logs


def configure_optimizers(args, model, cur_iter=-1):
    iters = args.iters

    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if args.opt == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': args.weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]

    LR = args.lr

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=LR,
            momentum=0.9,
        )
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=LR,
        )
    elif args.opt == 'lars':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=LR,
            momentum=0.9,
        )
        larc_optimizer = LARS(optimizer)
    else:
        raise NotImplementedError

    if args.lr_schedule == 'warmup-anneal':
        scheduler = utils.LinearWarmupAndCosineAnneal(
            optimizer,
            args.warmup,
            iters,
            last_epoch=cur_iter,
        )
    elif args.lr_schedule == 'linear':
        scheduler = utils.LinearLR(optimizer, iters, last_epoch=cur_iter)
    elif args.lr_schedule == 'const':
        scheduler = None
    else:
        raise NotImplementedError

    if args.opt == 'lars':
        optimizer = larc_optimizer

    # if args.verbose:
    #     print('Optimizer : ', optimizer)
    #     print('Scheduler : ', scheduler)

    return optimizer, scheduler
