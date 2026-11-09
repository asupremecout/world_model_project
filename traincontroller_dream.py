"""
Train a linear controller IN THE DREAM (VAE decoder + MDN-RNN simulated env).

Mirror of traincontroller.py, but the rollout happens entirely in latent space
(no real CarRacing / Box2D), using the MDRNN's predicted reward.
The resulting controller can then be transferred to the real env for evaluation
(the standard World Models protocol).

Usage:
    python traincontroller_dream.py --logdir exp_dream --max-generations 100 \
        --pop-size 32 --n-samples 4 --max-workers 16
"""
import argparse
import sys
from os.path import join, exists
from os import makedirs, listdir, unlink, getpid
from time import sleep
from torch.multiprocessing import Process, Queue
import torch
from torch.distributions.categorical import Categorical
import cma
from models import Controller
from models.vae import VAE
from models.mdrnn import MDRNNCell
from tqdm import tqdm
import numpy as np
from utils.misc import ASIZE, RSIZE, LSIZE
from utils.misc import load_parameters, flatten_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where everything is stored.')
parser.add_argument('--n-samples', type=int, default=4)
parser.add_argument('--pop-size', type=int, default=32)
parser.add_argument('--max-generations', type=int, default=100)
parser.add_argument('--max-workers', type=int, default=16)
parser.add_argument('--time-limit', type=int, default=1000)
parser.add_argument('--display', action='store_true', help='Show tqdm progress bar.')
parser.add_argument('--log-step', type=int, default=5)
args = parser.parse_args()

n_samples = args.n_samples
pop_size = args.pop_size
num_workers = min(args.max_workers, n_samples * pop_size)
time_limit = args.time_limit
log_step = args.log_step

tmp_dir = join(args.logdir, 'tmp_dream')
ctrl_dir = join(args.logdir, 'ctrl')
ctrl_file = join(ctrl_dir, 'best.tar')

if __name__ == '__main__':
    makedirs(tmp_dir, exist_ok=True)
    for fname in listdir(tmp_dir):
        unlink(join(tmp_dir, fname))
    makedirs(ctrl_dir, exist_ok=True)


def make_generator(mdir, device, tlimit):
    """Build VAE + MDRNN + controller for dream rollouts."""
    vae_file = join(mdir, 'vae', 'best.tar')
    rnn_file = join(mdir, 'mdrnn', 'best.tar')
    assert exists(vae_file) and exists(rnn_file), "missing vae/mdrnn in %s" % mdir
    mloc = {'cuda:0': str(device)} if device.type == 'cuda' else 'cpu'
    vae_state = torch.load(vae_file, map_location=mloc)
    rnn_state = torch.load(rnn_file, map_location=mloc)

    vae = VAE(3, LSIZE).to(device)
    vae.load_state_dict(vae_state['state_dict'])
    vae.eval()

    mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
    mdrnn.load_state_dict({k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
    mdrnn.eval()

    controller = Controller(LSIZE, RSIZE, ASIZE).to(device)
    return vae, mdrnn, controller


def dream_rollout(controller, mdrnn, params, device, tlimit):
    """Rollout in the dream; return minus cumulative predicted reward."""
    if params is not None:
        load_parameters(params, controller)
    with torch.no_grad():
        latent = torch.randn(1, LSIZE).to(device)
        hidden = [torch.zeros(1, RSIZE).to(device) for _ in range(2)]
        cumulative = 0.0
        for _ in range(tlimit):
            action = controller(latent, hidden[0])
            mu, sigma, logpi, r, d, hidden = mdrnn(action, latent, hidden)
            cumulative += r.item()
            mixt = Categorical(torch.exp(logpi.squeeze())).sample().item()
            latent = mu[:, mixt, :]
    return -cumulative


def slave_routine(p_queue, r_queue, e_queue, p_index):
    gpu = p_index % torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
    sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')
    vae, mdrnn, controller = make_generator(args.logdir, device, time_limit)
    while e_queue.empty():
        if p_queue.empty():
            sleep(.1)
        else:
            s_id, params = p_queue.get()
            r_queue.put((s_id, dream_rollout(controller, mdrnn, params, device, time_limit)))


p_queue = Queue()
r_queue = Queue()
e_queue = Queue()


def main():
    for p_index in range(num_workers):
        Process(target=slave_routine, args=(p_queue, r_queue, e_queue, p_index)).start()

    controller = Controller(LSIZE, RSIZE, ASIZE)
    cur_best = None
    if exists(ctrl_file):
        state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
        cur_best = -state['reward']
        controller.load_state_dict(state['state_dict'])
        print("Loaded previous best dream score {}".format(-cur_best))

    es = cma.CMAEvolutionStrategy(flatten_parameters(controller.parameters()), 0.1,
                                  {'popsize': pop_size})

    for gen in range(args.max_generations):
        r_list = [0.0] * pop_size
        solutions = es.ask()
        for s_id, s in enumerate(solutions):
            for _ in range(n_samples):
                p_queue.put((s_id, s))

        pbar = tqdm(total=pop_size * n_samples) if args.display else None
        for _ in range(pop_size * n_samples):
            while r_queue.empty():
                sleep(.05)
            r_s_id, r = r_queue.get()
            r_list[r_s_id] += r / n_samples
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

        es.tell(solutions, r_list)
        es.disp()

        best_idx = int(np.argmin(r_list))
        if cur_best is None or r_list[best_idx] < cur_best:
            cur_best = r_list[best_idx]
            best_params = solutions[best_idx]
            load_parameters(best_params, controller)
            torch.save({'epoch': gen, 'reward': -cur_best,
                        'state_dict': controller.state_dict()}, ctrl_file)
            print("[dream] gen %d  new best dream score %.2f" % (gen, -cur_best))

    es.result_pretty()
    e_queue.put('EOP')


if __name__ == '__main__':
    main()
