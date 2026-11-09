"""
Collect rollouts with a TRAINED controller (expert data), used to train a
better reward model for dream training.

Mirrors the action computation of utils.misc.RolloutGenerator, but saves the
full (observations, actions, rewards, terminals) trajectory like data/carracing.py.
"""
import argparse
import numpy as np
import torch
from os.path import join, exists
from os import makedirs
import gym
import gym.envs.box2d
from models import VAE, MDRNNCell, Controller
from utils.misc import LSIZE, ASIZE, RSIZE, transform

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='exp1')
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--rollouts', type=int, default=32)
parser.add_argument('--time-limit', type=int, default=1000)
args = parser.parse_args()


def make_models(logdir, device):
    mloc = {'cuda:0': str(device)} if device.type == 'cuda' else 'cpu'
    vae = VAE(3, LSIZE).to(device)
    vae.load_state_dict(torch.load(join(logdir, 'vae', 'best.tar'), map_location=mloc)['state_dict'])
    vae.eval()
    mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
    mdrnn.load_state_dict({k.strip('_l0'): v for k, v in
                           torch.load(join(logdir, 'mdrnn', 'best.tar'), map_location=mloc)['state_dict'].items()})
    mdrnn.eval()
    ctrl = Controller(LSIZE, RSIZE, ASIZE).to(device)
    ctrl.load_state_dict(torch.load(join(logdir, 'ctrl', 'best.tar'), map_location=mloc)['state_dict'])
    ctrl.eval()
    return vae, mdrnn, ctrl


def main():
    assert exists(args.dir), "The data directory does not exist..."
    device = torch.device('cuda')
    vae, mdrnn, ctrl = make_models(args.logdir, device)
    env = gym.make('CarRacing-v0')
    for i in range(args.rollouts):
        obs = env.reset()
        env.render()  # required first render
        hidden = [torch.zeros(1, RSIZE).to(device) for _ in range(2)]
        S, A, R, D = [], [], [], []
        t = 0
        while True:
            ot = transform(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                _, mu, _ = vae(ot)
                act = ctrl(mu, hidden[0])
                _, _, _, _, _, hidden = mdrnn(act, mu, hidden)
            a = act.squeeze().cpu().numpy()
            obs, r, done, _ = env.step(a)
            S.append(obs)
            A.append(a)
            R.append(r)
            D.append(done)
            t += 1
            if done or t >= args.time_limit:
                break
        np.savez(join(args.dir, 'rollout_{}'.format(i)),
                 observations=np.array(S), actions=np.array(A),
                 rewards=np.array(R), terminals=np.array(D))
        print('rollout {} frames {} return {:.1f}'.format(i, t, sum(R)), flush=True)


if __name__ == '__main__':
    main()
