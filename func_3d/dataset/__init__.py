from .btcv import BTCV
from .amos import AMOS
from .combined import Combined
from .sarcoma import Sarcoma
from .msd import MSD
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Subset


def _build_loader(dataset, shuffle, num_workers, rank=None, world_size=None, distributed=False):
    if distributed:
        # No drop_last on the eval splits: dropping the tail would silently score the
        # model on fewer volumes than the split contains.
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,
                          pin_memory=True, sampler=sampler)

    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def get_dataloader(args, rank=None, world_size=None, splits=("train", "val", "test")):
    """Return (train, val, test) loaders; any split not named in `splits` comes back None.

    `val` is carved out of the training manifest by patient group, following the fold
    assignment split_fold.py committed under data/splits (see
    func_3d/dataset/splits.py), and is what model selection should use; `test` is the
    held-out manifest. `val` is None when -fold < 0, which trains on the whole training
    manifest and leaves nothing to select on.

    `splits` is what keeps the test set out of a training run: train_3d.py asks for
    ("train", "val") only, so the *Ts.csv manifests are never even opened while
    training. Scoring the test split is eval_3d.py's job alone.
    """
    unknown = set(splits) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unknown split(s) {sorted(unknown)}; expected any of 'train', 'val', 'test'")

    want_train = "train" in splits
    want_val = "val" in splits and args.fold >= 0
    want_test = "test" in splits

    if args.dataset == 'combined': #nii
        build = lambda mode: Combined(args, mode=mode)
        workers = (4, 4)
    elif args.dataset == 'amos':
        '''amos data'''
        # AMOS predates the manifest CSVs and has no grouped split yet, so it has no
        # val fold to hold out -- an -eval_split val run on it has nothing to score.
        amos_kwargs = dict(transform=None, transform_msk=None, prompt=args.prompt)
        nice_train_loader = DataLoader(
            AMOS(args, args.data_path, mode='Training', **amos_kwargs),
            batch_size=1, shuffle=True, num_workers=8, pin_memory=True,
        ) if want_train else None
        nice_test_loader = DataLoader(
            AMOS(args, args.data_path, mode='Test', **amos_kwargs),
            batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
        ) if want_test else None
        return nice_train_loader, None, nice_test_loader
        '''end'''
    elif args.dataset == 'sarcoma':
        build = lambda mode: Sarcoma(args, subset=mode)
        workers = (4, 4)
    elif args.dataset == 'msd':
        build = lambda mode: MSD(args, mode=mode)
        workers = (4, 2)
    elif args.dataset == 'btcv': #png
        '''btcv data'''
        build = lambda mode: BTCV(args, subset=mode)
        workers = (2, 2)
    else:
        raise ValueError(f"the dataset {args.dataset} is not supported now!!!")

    train_workers, eval_workers = workers
    loader_kwargs = dict(rank=rank, world_size=world_size, distributed=args.distributed)

    nice_train_loader = (
        _build_loader(build("train"), shuffle=True, num_workers=train_workers, **loader_kwargs)
        if want_train else None
    )
    nice_val_loader = (
        _build_loader(build("val"), shuffle=False, num_workers=eval_workers, **loader_kwargs)
        if want_val else None
    )
    nice_test_loader = (
        _build_loader(build("test"), shuffle=False, num_workers=eval_workers, **loader_kwargs)
        if want_test else None
    )

    return nice_train_loader, nice_val_loader, nice_test_loader
