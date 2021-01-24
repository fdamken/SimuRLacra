import numpy as np
import torch
from numba import jit, typeof
from numba import int64, int32, float32
from numba.experimental import jitclass


class Buffer:
    """
    The Buffer stores all relevant information about the sampled trajectoires as numpy arrays.
    """

    def __init__(self, obs_dim: np.array, act_dim: np.array, size: int, gamma: float, lam: float, num: int):
        """
        Constructor

        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param size: maximum trajectory size
        :param gamma: discount factor
        :param lam: lambda factor for GAE
        :param num: number of environments for parallel sampling
        """

        self.obs_buf = np.empty(
            (
                num,
                size,
            )
            + obs_dim,
            dtype=np.float32,
        )
        self.act_buf = np.empty((num, size) + act_dim, dtype=np.float32)
        self.rew_buf = np.empty((num, size), dtype=np.float32)
        self.val_buf = np.empty((num, size), dtype=np.float32)
        self.ret_buf = np.zeros((num, size), dtype=np.float32)
        self.adv_buf = np.zeros((num, size), dtype=np.float32)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.ptr = 0
        self.sections = [[0] for _ in range(num)]
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.num = num

    def reset(self):
        """ Resets the intern memory. """
        self.ret_buf = np.zeros((self.num, self.size), dtype=np.float32)
        self.adv_buf = np.zeros((self.num, self.size), dtype=np.float32)

        self.ptr = 0
        self.sections = [[0] for _ in range(self.num)]

    def store(self, obs: np.array, act: np.array, rew: np.array, val: np.array):
        """
        Stores a set of information for a single timestep.

        :param obs: observation
        :param act: action
        :param rew: reward
        :param val: value
        """
        assert self.ptr < self.size, "Buffer full!"

        self.obs_buf[:, self.ptr] = obs
        self.act_buf[:, self.ptr] = act
        self.rew_buf[:, self.ptr] = rew
        self.val_buf[:, self.ptr] = val
        self.ptr += 1

    def avg_rew(self):
        """ Calculates the average reward achieved. """
        avg_rews = np.zeros(self.num)
        for n in range(self.num):
            sections_num = len(self.sections[n]) - 1
            if sections_num == 0:
                avg_rews[n] = self.rew_buf[n].sum()
            else:
                avg_rews[n] = self.rew_buf[n][: self.sections[n][sections_num]].sum() / sections_num
        return avg_rews.mean()

    def ret_and_adv(self):
        """ Calculates the return and advantages. """
        for n in range(self.num):
            for i in range(len(self.sections[n]) - 1):
                start, end = self.sections[n][i], self.sections[n][i + 1]

                self.ret_buf[n][start:end] = game_ret(end - start, self.rew_buf[n][start:end], self.gamma)
                self.adv_buf[n][start:end] = adv_estimation(
                    end - start, self.rew_buf[n][start:end], self.val_buf[n][start:end], False, self.gamma, self.lam
                )

            if self.sections[n][-1] != self.size:
                start = self.sections[n][-1]
                tmp = self.rew_buf[n][-1]
                self.rew_buf[n][-1] = self.val_buf[n][-1]

                self.ret_buf[n][start:] = game_ret(self.size - start, self.rew_buf[n][start:], self.gamma)
                self.rew_buf[n][-1] = tmp
                self.adv_buf[n][start:-1] = adv_estimation(
                    self.size - start, self.rew_buf[n][start:], self.val_buf[n][start:], True, self.gamma, self.lam
                )

    def get_data(self):
        """ Get all valid data from the buffers. """
        obs = self.obs_buf[: self.ptr].reshape((-1,) + self.obs_dim)
        act = self.act_buf[: self.ptr].reshape((-1,) + self.act_dim)
        rew = self.rew_buf[: self.ptr].reshape(-1)
        ret = self.ret_buf[: self.ptr].reshape(-1)
        adv = self.adv_buf[: self.ptr].reshape(-1)

        return obs, act, rew, ret, adv


# Functions are outside of class to use numba


@jit(nopython=True)
def game_ret(len: int, rew_buf: np.array, gamma: float):
    """
    Fast return calculation using numba.

    :param len: length of trajectory
    :param rew_buf: reward buffer
    :param gamma: discount factor
    """
    ret_buf = np.zeros(len, dtype=np.float32)
    for k in range(len):
        ret_buf[: len - k] += (gamma ** k) * rew_buf[k:]
    return ret_buf


@jit(nopython=True)
def adv_estimation(len: int, rews: np.array, vals: np.array, is_end: bool, gamma: float, lam: float):
    """
    Fast advantage estimation using GAE and numba.

    :param len: length of trajectory
    :param rews: rewards
    :param rews: values
    :param is_end: is last trajectory?
    :param gamma: discount factor
    :param lam: lambda factor for GAE
    """
    if is_end:
        rews = rews[:-1]
        len -= 1
    else:
        vals = np.concatenate((vals, np.zeros(1, dtype=np.float32)), axis=0)

    adv_buf = np.zeros(len, dtype=np.float32)

    deltas = rews + gamma * vals[1:] - vals[:-1]

    for l in range(len):
        adv_buf[: len - l] += (lam * gamma) ** l * deltas[l:]

    return adv_buf
