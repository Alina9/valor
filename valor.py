import numpy as np
import pybullet_envs
import torch
import torch.nn.functional as F 
import gym
from collections import deque
from network import Decoder, ActorCritic
from buffer import Buffer
from torch.distributions.categorical import Categorical


def valor(env, actor_critic=ActorCritic, ac_kwargs=dict(), decoder=Decoder, dc_kwargs=dict(), seed=0, episodes_per_epoch=100,
          epochs=1000, gamma=0.99, pi_lr=3e-3, vf_lr=1e-3, dc_lr=8e-3, train_v_iters=80, train_dc_iters=10, train_dc_interv=1,
          lam=0.99, max_ep_len=1000, con_dim=5, k=1e-1):


    seed += 10000
    torch.manual_seed(seed)
    np.random.seed(seed)

    #env
    state_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    ac_kwargs['action_space'] = env.action_space

    # Model
    actor_critic = actor_critic(input_dim=state_dim[0]+con_dim, **ac_kwargs)
    decoder = decoder(input_dim=state_dim[0], context_dim=con_dim, **dc_kwargs)

    # Buffer
    buffer = Buffer(con_dim, state_dim[0], act_dim, episodes_per_epoch, max_ep_len, train_dc_interv, gamma, lam)

    # Optimizers
    train_pi = torch.optim.Adam(actor_critic.policy.parameters(), lr=pi_lr)
    train_v = torch.optim.Adam(actor_critic.value_f.parameters(), lr=vf_lr)
    train_dc = torch.optim.Adam(decoder.policy.parameters(), lr=dc_lr)


    def update(e):
        obs, act, adv, pos, ret, logp_old = [torch.Tensor(x) for x in buffer.retrieve_all()]
        
        # Policy
        _, logp, _ = actor_critic.policy(obs, act)
        entropy = (-logp).mean()

        # Policy loss
        pi_loss = -(logp*(k*adv+pos)).mean() - 1e-3*entropy

        # Train policy
        train_pi.zero_grad()
        pi_loss.backward()
        train_pi.step()

        # Value function
        for _ in range(train_v_iters):
            v = actor_critic.value_f(obs)
            v_loss = F.mse_loss(v, ret)

            # Value function train
            train_v.zero_grad()
            v_loss.backward()
            train_v.step()

        # Discriminator
        if (e+1) % train_dc_interv == 0:
            print('Discriminator Update!')
            con, s_diff = [torch.Tensor(x) for x in buffer.retrieve_dc_buff()]

            # Discriminator train
            for _ in range(train_dc_iters):
                _, logp_dc, _ = decoder(s_diff, con)
                d_loss = -logp_dc.mean()
                train_dc.zero_grad()
                d_loss.backward()
                train_dc.step()


    state, reward, done, ep_reward, ep_len = env.reset(), 0, False, 0, 0
    context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))
    total_t = 0
    ep = 0
    skills= deque(maxlen=100)
    for epoch in range(epochs):
        actor_critic.eval()
        decoder.eval()
        for _ in range(episodes_per_epoch):
            ep += 1
            c = context_dist.sample()
            c_onehot = F.one_hot(c, con_dim).squeeze().float()
            step = 0
            for _ in range(max_ep_len):
                step+=1
                concat_state = torch.cat([torch.Tensor(state.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)
                action, _, logp_t, v_t = actor_critic(concat_state)
                buffer.store(c, concat_state.squeeze().detach().numpy(), action.detach().numpy(), reward, v_t.item(), logp_t.detach().numpy())

                state, reward, done, _ = env.step(action.detach().numpy()[0])
                ep_reward += reward
                ep_len += 1
                total_t += 1
                terminal = done or (ep_len == max_ep_len)
                if terminal:
                    dc_diff = torch.Tensor(buffer.calc_diff()).unsqueeze(0)
                    context = torch.Tensor([float(c)]).unsqueeze(0)
                    label, logq, log_p = decoder(dc_diff, context)
                    skills.append(logq.item())
                    buffer.end_episode(logq.detach().numpy())
                    print(
                        f'{ep}) steps:{step}, Episode skill reward: {logq.item()}, average skill reward: '
                        f'{np.mean(skills)}, context: {int(context.item())}, label: {label.item()}, reward: {ep_reward}')

                    state, reward, done, ep_reward, ep_len = env.reset(), 0, False, 0, 0

                    break
        # Update
        actor_critic.train()
        decoder.train()

        update(epoch)

        if ep % 100 == 0:
            torch.save(actor_critic.state_dict(), f"agent_critic{ep}.pickle")
            torch.save(decoder.state_dict(), f"disc{ep}.pickle")


if __name__ == '__main__':
    #env = gym.make('MountainCarContinuous-v0')
    env = gym.make("HalfCheetahBulletEnv-v0")





    valor(env, actor_critic=ActorCritic, ac_kwargs=dict(hidden_dims=[64,64]),
          decoder=Decoder, dc_kwargs=dict(hidden_dims=64),
          gamma=0.99, seed=0, episodes_per_epoch=100, epochs=1000, con_dim=5)

