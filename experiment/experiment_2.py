import argparse
from email import utils
import pandas as pd

import utils.utils as utils
from utils.custom_scheduler import CosineAnnealingWarmUpRestarts
from utils.preprocessing import load_data
from utils.inference import inference_tabnet
from torch.optim import Adam


if '__name__' == '__main__':
    def parser():
        parser = argparse.ArgumentParser(description='tabnet experiment')
        parser.add_argument('--train', type=str, default='./train.csv')
        parser.add_argument('--test', type=str, default='./test.csv')
        parser.add_argument('--sub', type=str, default='./submssion.csv')
        parser.add_argument('--seed', type = int, default=42)
        parser.add_argument('--n_d',type=int, default=64)
        parser.add_argument('--n_step',type=int, default=3)
        parser.add_argument('--n_independent',type=int, default=2)
        parser.add_argument('--n_gamma',type=float, default=1.3)
        parser.add_argument('--n_shared',type=int, default=3)
        parser.add_argument('--learning_rate', type=float, default=4e-5)
        parser.add_argument('--weight_decay', type=float, default=5e-7)
        parser.add_argument('--t0', type=int, default=5)
        parser.add_argument('--tmult', type=int, default=2)
        parser.add_argument('--eta_max', type=float, default=7e-8)
        parser.add_argument('--tup', type=int, default=3)
        parser.add_argument('--gamma',type=float, default=4e-5)
        args = parser.parse_args()

        return args

    args = parser()
    utils.seed_everything(args.seed)
    sub = pd.read_csv(args.sub)

    tabnet_params = dict(
        n_d = args.n_d,
        n_a = args.n_d,
        n_steps = args.n_step,
        gamma = args.n_gamma,
        n_independent = args.n_independent,
        n_shared = args.n_shared,
        optimizer_fn = Adam,
        optimizer_params = dict(lr = (args.learning_rate), weight_decay = args.weight_decay),
        mask_type = "entmax",
        scheduler_params = dict(T_0 = args.t0, T_mult = args.tmult, eta_max = args.eta_max, T_up = args.tup, gamma = args.gamma),
        scheduler_fn = CosineAnnealingWarmUpRestarts,
        seed = args.seed,
    )

    train_x, train_y, test = load_data(args.train, args.test)
    test_predict = inference_tabnet(train_x, train_y, test, params=tabnet_params, args=args)

    sub.iloc[:, 1:] = test_predict
    sub.to_csv('./tabnet.csv', index=False)