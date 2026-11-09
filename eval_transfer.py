import argparse
import numpy as np
import torch
from utils.misc import RolloutGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, required=True)
parser.add_argument('--n', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda')
gen = RolloutGenerator(args.logdir, device, 1000)
rets = []
for k in range(args.n):
    np.random.seed(args.seed + k)  # 固定每条赛道，保证 baseline / 改进版可比
    with torch.no_grad():
        cum = -gen.rollout(None)
    rets.append(cum)
    print('track %d  return %.1f' % (k, cum), flush=True)
rets = np.array(rets)
print('MEAN %.1f  STD %.1f  MAX %.1f  MIN %.1f' % (rets.mean(), rets.std(), rets.max(), rets.min()))
