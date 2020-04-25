import numpy as np
import scipy.signal



class Buffer(object):
    def __init__(self, con_dim, state_dim, act_dim, batch_size, ep_len, dc_interv, gamma=0.99, lam=0.95):
        self.max_batch = batch_size
        self.dc_interv = dc_interv
        self.max_s = batch_size * ep_len
        self.state_dim = state_dim
        self.st = np.zeros((self.max_s, state_dim + con_dim))
        self.act = np.zeros((self.max_s, act_dim))
        self.rew = np.zeros(self.max_s)
        self.ret = np.zeros(self.max_s)
        self.adv = np.zeros(self.max_s)
        self.log_q = np.zeros(self.max_s)
        self.logp = np.zeros(self.max_s)
        self.val = np.zeros(self.max_s)
        self.end = np.zeros(batch_size + 1)
        self.position = 0
        self.eps = 0
        self.dc_eps = 0

        self.N = 11

        self.con = np.zeros(self.max_batch * self.dc_interv)
        self.dcbuf = np.zeros((self.max_batch * self.dc_interv, self.N - 1, state_dim))

        self.gamma = gamma
        self.lam = lam

    def store(self, context, state, action, reward, value, logp):
        assert self.position < self.max_s
        self.st[self.position] = state
        self.act[self.position] = action
        self.con[self.dc_eps] = context
        self.rew[self.position] = reward
        self.val[self.position] = value
        self.logp[self.position] = logp
        self.position += 1

    def calc_diff(self):
        start = int(self.end[self.eps])
        ep_l = self.position - start - 1
        for i in range(self.N - 1):
            prev = int(i * ep_l / (self.N - 1))
            succ = int((i + 1) * ep_l / (self.N - 1))
            self.dcbuf[self.dc_eps, i] = self.st[start + succ][:self.state_dim] - self.st[start + prev][:self.state_dim]
        return self.dcbuf[self.dc_eps]

    def end_episode(self, log_prob, last_val=0):
        ep_slice = slice(int(self.end[self.eps]), self.position)
        rewards = np.append(self.rew[ep_slice], last_val)
        values = np.append(self.val[ep_slice], last_val)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        returns = scipy.signal.lfilter([1], [1, float(-self.gamma)], rewards[::-1], axis=0)[::-1]
        self.ret[ep_slice] = returns[:-1]
        self.adv[ep_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0)[::-1]
        self.log_q[ep_slice] = log_prob

        self.eps += 1
        self.dc_eps += 1
        self.end[self.eps] = self.position

    def retrieve_all(self):
        assert self.eps == self.max_batch
        cur_slice = slice(0, self.position)
        self.position = 0
        self.eps = 0
        adv_mean, adv_std = self.statistics_scalar(self.adv[cur_slice])
        logq_mean, pos_std = self.statistics_scalar(self.log_q[cur_slice])
        self.adv[cur_slice] = (self.adv[cur_slice] - adv_mean) / adv_std
        self.log_q[cur_slice] = (self.log_q[cur_slice] - logq_mean) / pos_std
        return [self.st[cur_slice], self.act[cur_slice], self.adv[cur_slice], self.log_q[cur_slice],
                self.ret[cur_slice], self.logp[cur_slice]]

    @staticmethod
    def statistics_scalar(array):
        return array.mean(), array.std()

    def retrieve_dc_buff(self):
        assert self.dc_eps == self.max_batch * self.dc_interv
        self.dc_eps = 0
        return [self.con, self.dcbuf]
