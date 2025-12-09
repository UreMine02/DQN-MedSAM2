from cfg import parse_args
from func_3d.dataset.btcv import BTCV
from torch.utils.data import DataLoader

args = parse_args()
btcv_train_dataset = BTCV(args, mode='train')
btcv_test_dataset = BTCV(args, mode='test')

nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

for data in nice_train_loader:
    pass