from my_agent_rho_reward import MyAgent

from grid2op import make
from lightsim2grid.LightSimBackend import LightSimBackend

from tqdm import trange
import pandas as pd
import numpy as np
import sys
import pickle

print('Running...')
tot_episodes    = 2500
total_timesteps = 5e5
batch_size      = 32
discount        = 0.99
policy_freq     = 10 
update_freq     = 1
print_update    = 2000
learning_start  = 100
e_episodes = tot_episodes
e_start     = 0.95
e_end       = 0.05
e_steps     = (e_start-e_end)/tot_episodes

train = True
load  = False

if train:
    print("Training")
else:
    print("Testing")

logfile = open('Outputs/logfile.log', 'w')
old_stdout = sys.stdout
sys.stdout = logfile


instance = "l2rpn_neurips_2020_track1_small"
BACKEND = LightSimBackend
env = make(instance, backend=BACKEND())
my_agent = MyAgent(env, '')
# replay_buffer_dqn = my_agent.replay_buffer_dqn()

if load:
    my_agent.load('')

# done = False
# reward = env.reward_range[0]

# def epsilon_decay(steps):
#     return e_end + (e_start - e_end) * math.exp(-1. * steps / e_steps)

if train:
    current_step_num = 0
    alive_steps = 0
    epsilon = e_start

    losses_per_episode_ddpg = []
    losses_per_episode_dqn  = []
    losses_per_episode_lr   = []

    reward_per_episode   = []
    survival_per_episode = []

    seeds = np.random.choice(np.arange(tot_episodes,
                                       10*tot_episodes),
                             size=tot_episodes,
                             replace=False)

    for eps in trange(tot_episodes, desc='Episode', leave=False):

        print(eps)

        losses, losses_dqn, losses_PolicyNet, losses_dqn_redispatch= [],[],[],[]
        survival_steps = 0
        episode_reward = 0
        done = False

        ep_state_pool = []
        ep_action_pool = []
        ep_reward_pool = []

        env = make(instance,  backend=BACKEND())
        env.seed(seeds[eps])
        obs = env.reset()
        my_agent.reset(obs)
        state = my_agent.utils.convert_obs(obs)

        while not done:

            print(survival_steps)

            action, action_rb, features = my_agent.getAction(obs, epsilon,train)

            new_obs, reward, done, info = env.step(action)
            new_state = my_agent.utils.convert_obs(new_obs)

            if done and len(info['exception'])!=0:
                reward = -1
            if info["is_illegal"] is True:
                reward = -1
            episode_reward += reward

            # pushes the (s, a, r, s') tuple onto the memory
            action_types = action_rb.keys()
            for action_type in action_types:
                print("action_type",action_type)

                if action_type == 'topo_ddpg':
                    actions = action_rb['topo_ddpg']
                    if done:
                        my_agent.replay_buffer_ddpg[actions[1]].add((np.array(state),
                                                                     actions[0], reward,
                                                                     np.array(new_state),
                                                                     1))

                    else:
                        my_agent.replay_buffer_ddpg[actions[1]].add((np.array(state),
                                                                  actions[0], reward,
                                                                  np.array(new_state),
                                                                  0))

                if action_type == 'topo_dqn':
                    actions = action_rb['topo_dqn']
                    print(actions)
                    my_agent.replay_buffer_dqn.add((np.array(state),
                                                    actions, reward,
                                                    np.array(new_state),
                                                    done))

                if action_type == 'disconnect':
                    action = action_rb['disconnect']
                    my_agent.lineModel.addToBuffer(action, reward)

                current_step_num +=1

            # updates the current state to be the
            # state receives from the environment
            obs = new_obs
            state = my_agent.utils.convert_obs(obs)
            alive_steps += 1
            survival_steps += 1
            # Decay chance of random action, update the step,
            # using function for epsilon decay
        if eps % update_freq == 0 and current_step_num > learning_start:

            print("\n")
            print("Topo Training Start")

            losses.append(my_agent.train(current_step_num,
                                         batch_size, discount,
                                         policy_freq))


            losses_dqn.append(my_agent.train_dqn(current_step_num,
                                                 batch_size, discount,
                                                 policy_freq))

            losses_dqn_redispatch.append(my_agent.train_redispatch_dqn(current_step_num,
                                                 batch_size, discount,
                                                 policy_freq))



            print("Topo Training Done")
            print("\n")

            #training of dqn and line reconnection
            my_agent.save()


        print("\n")
        print("Line Training Start")
        losses_PolicyNet.append(my_agent.update_PolicyNet())
        print("Line Training Done")
        print("\n")

        if eps < e_episodes:
            epsilon -= e_steps #0.0085
        else:
            epsilon = 0

        # updates model parameters every update_freq steps
        # and only after learning_start steps

        losses_per_episode_ddpg.append(np.sum(losses))
        losses_per_episode_dqn.append(np.sum(losses_dqn))
        losses_per_episode_lr.append(np.sum(losses_PolicyNet))

        reward_per_episode.append(episode_reward)
        survival_per_episode.append(survival_steps)


        pickle_in = open("Outputs/survival_steps.pickle","wb")
        pickle_in_2 = open("Outputs/rewards_steps.pickle","wb")
        pickle.dump(survival_per_episode, pickle_in)
        pickle.dump(reward_per_episode, pickle_in_2)

    # print("Training Done")
    sys.stdout.flush()
    train_log = (losses_per_episode_ddpg, losses_per_episode_dqn,
                 losses_per_episode_lr, reward_per_episode,
                 survival_per_episode)
    train_log = np.array(train_log).reshape(5,-1).transpose()
    pd.DataFrame(train_log).to_csv("Outputs/train_log.csv",
                  index=False, header=['losses_ddpg','losses_dqn',
                                       'losses_PolicyNet','rewards',
                                       'steps_survived'])


else:
    reward_per_episode    = []
    survival_per_episode  = []

    lrec_acts_per_episode = []
    ldis_acts_per_episode = []
    disp_acts_per_episode = []
    dqn_acts_per_episode  = []
    ddpg_acts_per_episode = []
    rec_acts_per_episode  = []

    seeds = np.arange(tot_episodes)

    for eps in trange(tot_episodes, desc='Episode', leave=False):

        num_sub16_acts = 0
        num_dqn_acts   = 0
        num_ldisc_acts = 0
        num_lrec_acts  = 0
        num_disp_acts  = 0
        num_rec_acts   = 0

        survival_steps = 0
        episode_reward = 0
        done = False

        env = make(instance)
        # env.seed(0)
        env.seed(seeds[eps])
        obs = env.reset()

        while not done:
            action, action_rb, _ = my_agent.getAction(obs,
                                                      epsilon=0,
                                                      train=train)
            new_obs, reward, done, info = env.step(action)

            action_types = action_rb.keys()
            for action_type in action_types:
                if action_type == 'topo_sub16':
                    num_sub16_acts += 1

                if action_type == 'topo_dqn':
                    num_dqn_acts += 1

                if action_type == 'disconnect':
                    num_ldisc_acts += 1

                if action_type == 'reconnect':
                    num_lrec_acts += 1

                if action_type == 'redispatch':
                    num_disp_acts += 1

                if action_type == 'recovery':
                    num_rec_acts += 1

            # pushes the (s, a, r, s') tuple onto the memory

            # updates the current state to be the state receives from the environment
            obs = new_obs

            # if len(topo_used):
            #     current_step_num +=1
            survival_steps += 1
            episode_reward += reward

        reward_per_episode.append(episode_reward)
        survival_per_episode.append(survival_steps)

        ddpg_acts_per_episode.append(num_sub16_acts)
        dqn_acts_per_episode.append(num_dqn_acts)
        ldis_acts_per_episode.append(num_ldisc_acts)
        lrec_acts_per_episode.append(num_lrec_acts)
        disp_acts_per_episode.append(num_disp_acts)
        rec_acts_per_episode.append(num_rec_acts)

    test_log = (seeds, reward_per_episode, survival_per_episode,
                ddpg_acts_per_episode, dqn_acts_per_episode,
                lrec_acts_per_episode, ldis_acts_per_episode,
                disp_acts_per_episode, rec_acts_per_episode)

    test_log = np.array(test_log).reshape(9,-1).transpose()
    pd.DataFrame(test_log).to_csv("Outputs/test_log.csv",
                 index=False, header=['seeds','rewards',
                                      'steps_survived',
                                      'topo_sub16_used',
                                      'topo_dqn_used',
                                      'line_rec_used',
                                      'line_disc_used',
                                      'redispatch_used',
                                      'recovery_used'])

sys.stdout = old_stdout
sys.stdout.flush()
logfile.flush()
logfile.close()
