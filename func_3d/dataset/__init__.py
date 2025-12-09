from .btcv import BTCV
from .amos import AMOS
from .combined import Combined
from .sarcoma import Sarcoma
from .msd import MSD
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Subset


def get_dataloader(args, rank=None, world_size=None):
    if args.dataset == 'combined': #nii
        combined_train_dataset = Combined(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        combined_test_dataset = Combined(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)
        
        if args.distributed:
            train_sampler = DistributedSampler(combined_train_dataset, num_replicas=world_size, rank=rank)
            test_sampler = DistributedSampler(combined_test_dataset, num_replicas=world_size, rank=rank)

            nice_train_loader = DataLoader(combined_train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)
            nice_test_loader = DataLoader(combined_test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=test_sampler)
        else:
            nice_train_loader = DataLoader(combined_train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
            nice_test_loader = DataLoader(combined_test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(amos_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'sarcoma':
        sarcom_train_dataset = Sarcoma(args, subset="train")
        sarcom_test_dataset = Sarcoma(args, subset="test")
        
        if args.distributed:
            train_sampler = DistributedSampler(sarcom_train_dataset, num_replicas=world_size, rank=rank)
            test_sampler = DistributedSampler(sarcom_test_dataset, num_replicas=world_size, rank=rank)

            nice_train_loader = DataLoader(sarcom_train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)
            nice_test_loader = DataLoader(sarcom_test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=test_sampler)
        else:
            nice_train_loader = DataLoader(sarcom_train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
            nice_test_loader = DataLoader(sarcom_test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        '''end'''
    elif args.dataset == 'msd':
        msd_train_dataset = MSD(args, mode="train")
        msd_test_dataset = MSD(args, mode="test")
        
        if args.distributed:
            train_sampler = DistributedSampler(msd_train_dataset, num_replicas=world_size, rank=rank)
            test_sampler = DistributedSampler(msd_test_dataset, num_replicas=world_size, rank=rank)

            nice_train_loader = DataLoader(msd_train_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, sampler=train_sampler)
            nice_test_loader = DataLoader(msd_test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, sampler=test_sampler)
        else:
            nice_train_loader = DataLoader(msd_train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
            nice_test_loader = DataLoader(msd_test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        '''end'''
    elif args.dataset == 'btcv': #png
        '''btcv data'''
        btcv_train_dataset = BTCV(args, subset='train')
        btcv_test_dataset = BTCV(args, subset='test')
        
        if args.distributed:
            train_sampler = DistributedSampler(btcv_train_dataset, num_replicas=world_size, rank=rank)
            test_sampler = DistributedSampler(btcv_test_dataset, num_replicas=world_size, rank=rank)

            nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)
            nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=test_sampler)
        else:
            nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
            nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        '''end'''
    else:
        raise ValueError(f"the dataset {args.dataset} is not supported now!!!")
        
    return nice_train_loader, nice_test_loader