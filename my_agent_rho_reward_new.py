import sys
import os, copy
import numpy as np
import pickle
import random
from grid2op.Agent import BaseAgent

# from .utils import utils, OUNoise, expBuffer # For Submission
# from .my_model import DDPG, DoubleDQN ,RedispatchModel # For Submission
# from .my_model import ReplayBuffer           # For Submission
# from .my_model import lineDisconnectionModel # For Submission
# from .my_model import AutoEncoder            # For Submission
# from .my_model import KNNTree                # For Submission

from utils import utils, OUNoise, expBuffer  # For Training
from my_model import DDPG, DoubleDQN,RedispatchModel         # For Training
from my_model import ReplayBuffer            # For Training
from my_model import lineDisconnectionModel  # For Training
from my_model import AutoEncoder             # For Training
from my_model import KNNTree                 # For Training


class MyAgent(BaseAgent):
    def __init__(self, env, dir_path):
        BaseAgent.__init__(self, env.action_space)  # action space converter

        self.obs_space = env.observation_space
        self.action_space = env.action_space

        # For Topology Agent

        self.nn               = 10
        self.batch_size       = 32
        self.rho_threshold    = 0.95
        self.action_size      = 17   # change 1 --> make it as the same length of topo vector
        self.num_comb_thresh  = 150
        self.topo_action_used = 0
        self.dqn_action_space = 78
        self.utils = utils(env)

        # self.done = False
        # self.start = True

        state_dim = (env.observation_space.n_gen +
                     env.observation_space.n_line * 8 +
                     env.observation_space.n_load +
                     env.observation_space.n_sub +
                     env.observation_space.dim_topo)

        sub_topo = pickle.load(open(os.path.join(dir_path,
                    'Inputs/all_sub_combomations_bus.pickle'), "rb"))
        num_of_combos = []
        for sub in sub_topo:
            num_of_combos.append(len(sub[1]))
        self.sub_id = np.where(np.array(num_of_combos) >= self.num_comb_thresh)[0]


        redispatch_actions_vec = pickle.load(open(os.path.join(dir_path,
                    'Inputs/redispatch_actions.pickle'), "rb")) # 25 redispatch
        self.redispatch_actions = []

        for i in range(redispatch_actions_vec.shape[0]):
            action = self.action_space.from_vect(redispatch_actions_vec[i])
            self.redispatch_actions.append(action)

        self.redispatch_months = set([3])
        self.used_combine_actions = False
        self.redispatch_cnt = 0
        self.max_redispatch_cnt = 3

        self.topo_combos = []
        self.kdtree      = []
        self.encoder     = []
        self.topoModel_DDPG  = []

        self.noise       = []
        self.replay_buffer_ddpg = []
        self.replay_buffer_dqn  = ReplayBuffer()
        self.replay_buffer_dqn_redispatch = ReplayBuffer()
        self.expBuffer = expBuffer()

        # self.replay_buffer_line_disconnection =


        self.sub_id = [16]
        #[16,21,23, ]

        # self.topoModel_DDPG = DDPG(state_dim, self.action_size, 10, dir_path)
        for sub in self.sub_id:
            self.topo_combos.append(sub_topo[sub][1])
            topo_4_model = np.array(sub_topo[sub][1], dtype='int')-1
            self.kdtree.append(KNNTree(topo_4_model))
            encoder = AutoEncoder(topo_4_model.shape[1], self.action_size)
            encoder.fit(topo_4_model)
            self.encoder.append(encoder)
            self.topoModel_DDPG.append((DDPG(state_dim,
                                       self.action_size,
                                       10, dir_path)))
            self.noise.append(OUNoise(1))
            self.replay_buffer_ddpg.append(ReplayBuffer())

        sys.stdout.flush()

        # For Disconnection Agent
        self.allowed_overflow_timestep = env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED
        self.num_lines = len(self.action_space.line_or_to_subid)
        self.gen_pmax  = self.action_space.gen_pmax.max()
        self.lineModel = lineDisconnectionModel(7, self.allowed_overflow_timestep)
        # self.lineModel.load()

        #DQN
        self.df = pickle.load(open(os.path.join(dir_path,
                    'Inputs/new_substation_topo_extended'), "rb"))

        self.sub_topo = pickle.load(open(os.path.join(dir_path,
                    'Inputs/sub_topo.pickle'), "rb"))

        action_dim = len(self.df)

        self.sub_neighbours_ =  pickle.load(open(os.path.join(dir_path,
                    'Inputs/substation_one_hop_info'), "rb"))

        self.sub_neigh_actions = pickle.load(open(os.path.join(dir_path,
                    'Inputs/q_value_neigh_sub'), "rb"))

        self.redispatchModel =  RedispatchModel(state_dim, 39, self.utils)

        self.topoModel = DoubleDQN(state_dim, action_dim,
                                   self.df, self.utils, self.sub_topo,
                                   self.sub_neighbours_, self.sub_neigh_actions)

        self.topo_action_sub_16 = pickle.load(open(os.path.join(dir_path,
                    'Inputs/sub_16_topo_vect'),"rb"))


        self.topo_action_sub_28 = pickle.load(open(os.path.join(dir_path,
                    'Inputs/sub_28_actions'),"rb"))


        self.topo_action_sub_23 = pickle.load(open(os.path.join(dir_path,
                    'Inputs/sub_23_actions'),"rb"))
         

    def save(self):
        # self.topoModel[-1].save(self.sub_id[-1])
        for idx, sub_id in enumerate(self.sub_id):
            self.topoModel_DDPG[idx].save(sub_id)

        self.topoModel.save()
        self.lineModel.save()
        self.redispatchModel.save()

    #save dqn
    def load(self, dir_path):
        for idx, sub_id in enumerate(self.sub_id):
            self.topoModel_DDPG[idx].load(sub_id)

        self.topoModel.load(dir_path)
        self.redispatchModel.load(dir_path)
        self.lineModel.load()

    def reset(self, obs):
        # self.noise.reset()
        self.lineModel.reset()

    def act(self, observation, reward=None, done=False):

        # Fallback Action - Do Nothing
        do_nothing_action = self.action_space({})
        act = do_nothing_action

        counter  = -1
        action_flag = {}
        all_actions = []
        sim_rho = []
        sim_rewards = []
        all_rew_acts = []
        non_ol_acts = []

        self.observation = observation
        
        disconnect_act  = None
        reconnect_act   = None
        reconnect_action = None
        recovery_action = None
        disconnect_action = None
        redispatch_action = None
        topo_dqn_act = None
        redispatch_act_buffer = None
        disco_topo_act = None
        reco_topo_act = []
        non_ol_reward = []

        present_rho =  np.sum(sorted((observation.rho), reverse=True)[:2])

        do_nothing_action = self.action_space({})
        sim_obs,sim_reward,sim_done,_ = observation.simulate(do_nothing_action)
        do_nothing_reward = sim_reward

        present_rho_threshold = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

        non_ol_acts.append(do_nothing_action)
        non_ol_reward.append(sim_reward)
        
        if present_rho_threshold <= present_rho:
            sim_rho.append(present_rho_threshold)
            all_actions.append(do_nothing_action)

        # Reconnect Action 
        act, line_id = self._reconnect_action()
        if act is not None:
            reconnect_act = line_id
            reconnect_action = act
            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

            non_ol_acts.append(act)
            non_ol_reward.append(sim_reward)
            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])
            if  act_rho <= present_rho_threshold:
                sim_rho.append(act_rho)
                all_actions.append(act)
        #recover actions if the observation is not None 
        if self.observation is not None and not any(np.isnan(self.observation.rho)):
            if np.all(self.observation.topo_vect != -1):


                act = self._reset_redispatch()
                recovery_redisp_act = act
                if recovery_redisp_act is not None:
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    if not sim_done:
                        act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                        non_ol_acts.append(act)
                        non_ol_reward.append(sim_reward)
                                
                        if act_rho <= present_rho_threshold:
                            sim_rho.append(act_rho)
                            all_actions.append(act)
                
                act = self._reset_topology()
                recovery_action = act

                if recovery_action is not None:
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    if not sim_done:
                        act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                        non_ol_acts.append(act)
                        non_ol_reward.append(sim_reward)
                                
                        if act_rho <= present_rho_threshold:
                            sim_rho.append(act_rho)
                            all_actions.append(act)                

# checking if rho value is greater than one then taking the topology action 

        ol_list = self.getRankedOverloadList(observation)

        if len(ol_list):
            # Line Disconnection Action  
            line_id = self.lineModel.select_Action(observation, self.num_lines, self.gen_pmax)
            if line_id is not None:
                act = self.action_space({"set_line_status": [(line_id,-1)]})
                disconnect_act = line_id
                disconnect_action = act
                sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
                act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                if act_rho <= present_rho_threshold:
                    sim_rho.append(act_rho)
                    all_actions.append(act)


            # Topological Action  and combined with line actions

            # converting current observation into the feature for NN model

            state = self.utils.convert_obs(observation)
            #redispatch action
            # do_nothing_action = self.action_space({})
            # sim_obs_do_nothing, sim_reward_do_nothing,sim_done_do_nothing, sim_info_do_nothing = observation.simulate(do_nothing_action)
            # act, _ = self.redispatchModel.select_action(observation,state)

            # #simulating the action to check the legal action 
            
            # sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

            # # #comparing the do nothing action with the redispatch action 
            # if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
            #     # if sim_reward > do_nothing_reward:
            #     #     sim_rewards.append(sim_reward)
            #     #     all_rew_acts.append(act)
            #     if np.max(sim_obs.rho) <= present_rho_threshold:
            #         sim_rho.append(np.max(sim_obs.rho))
            #         all_actions.append(act)
            #     else:
            #         act = do_nothing_action
            #         redispatch_act_buffer = None
            # else:
            #     act = do_nothing_action
            #     redispatch_act_buffer = None 
            
            # redispatch_action = act
            
            # sim_rewards.append(sim_reward)
            # all_rew_acts.append(act)

            #redispatch action from the first winner

            

        #change number 2 ---> can be used simulated observation instead of the policy
            # ddpg model for the larger substations

            for indx, sub_id in enumerate(self.sub_id):
                noise = self.noise[indx].noise()
                
                policy = self.topoModel_DDPG[indx].actor_act(state, self.kdtree[indx].comb_len, noise)                
                nn_idx = self.kdtree[indx].get_nn_topos(policy, self.nn)
                nn_actions = []

                for idx in nn_idx:
                    nn_actions.append(self.kdtree[indx].get_actions(idx))   

                nn_actions = np.array(nn_actions, dtype='int')
                enc_topos  = self.encoder[indx].encode(nn_actions).detach().cpu().numpy()
                enc_topos  = enc_topos.reshape(self.nn, self.action_size)                
                q_vals = self.topoModel_DDPG[indx].critic_act(state, enc_topos)                
                q_vals = q_vals.detach().cpu().numpy().flatten()                
                sorted_idx = np.argsort(q_vals)[::-1]
                q_indx = np.argmax(q_vals)                  # index of max_qval
                tp_idx = nn_idx[q_indx]                     # get actual topo index 

                # For topological action alone
                for q_indx in sorted_idx:             
                    tp_idx = nn_idx[q_indx]                    
                    
                    if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                        topo_ = self.topo_combos[indx][tp_idx]                        
                        topo_legal = self.legal_topo_vect(sub_id, topo_)
                        act = self.action_space({"set_bus":{"substations_id": [(sub_id, topo_legal)]}})
                        
                        sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
                        act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                        if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                            if not sim_done:
                                if act_rho <= present_rho_threshold:
                                    sim_rho.append(act_rho)
                                    all_actions.append(act)
                                    break 


                # combining the ddpg action and redispatch
                for q_indx in sorted_idx:             
                    tp_idx = nn_idx[q_indx]                    
                    
                    if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                        topo_ = self.topo_combos[indx][tp_idx]                        
                        topo_legal = self.legal_topo_vect(sub_id, topo_)
                        act = self.action_space({"set_bus":{"substations_id": [(sub_id, topo_legal)]}})
                        act = self.combine_with_redispatch(observation, act)
                        sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
                        act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                        if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                            if not sim_done:
                                if act_rho <= present_rho_threshold:
                                    sim_rho.append(act_rho)
                                    all_actions.append(act)
                                    break 


                # If reconnect action is present, combining it with the ddpg substation 16 action
                if reconnect_act is not None:                               
                    for q_indx in sorted_idx:             
                        tp_idx = nn_idx[q_indx]

                        if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                            topo_ = self.topo_combos[indx][tp_idx]                        
                            topo_legal = self.legal_topo_vect(sub_id, topo_)
                            act = self.combine_with_topology(reconnect_act, sub_id, topo_legal, 1)
                            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])
                        
                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)
                                        break

                    for q_indx in sorted_idx:             
                        tp_idx = nn_idx[q_indx]

                        if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                            topo_ = self.topo_combos[indx][tp_idx]                        
                            topo_legal = self.legal_topo_vect(sub_id, topo_)
                            act = self.combine_with_topology(reconnect_act, sub_id, topo_legal, 1)
                            
                            act = self.combine_with_redispatch(observation, act)
                            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])
                        
                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)
                                        break


                
                
                # If disconnect action is present, combining it with the disconnect action
                if disconnect_act is not None:                               
                    for q_indx in sorted_idx:             
                        tp_idx = nn_idx[q_indx]
                        
                        if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                            topo_ = self.topo_combos[indx][tp_idx]                        
                            topo_legal = self.legal_topo_vect(sub_id, topo_)
                            act = self.combine_with_topology(disconnect_act, sub_id, topo_legal, -1)
                            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])
                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(np.max(sim_obs.rho))
                                        all_actions.append(act)
                                        break

                    for q_indx in sorted_idx:             
                        tp_idx = nn_idx[q_indx]
                        
                        if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                            topo_ = self.topo_combos[indx][tp_idx]                        
                            topo_legal = self.legal_topo_vect(sub_id, topo_)
                            act = self.combine_with_topology(disconnect_act, sub_id, topo_legal, -1)
                            act = self.combine_with_redispatch(observation, act)

                            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])
                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(np.max(sim_obs.rho))
                                        all_actions.append(act)
                                        break                            

            #DQN actions             
            act, _, topo, sub_id = self.topoModel.select_action(state, observation, self.action_space)
                
            if (act != None and self.observation.time_before_cooldown_sub[sub_id] == 0): 
                topo_legal =  self.legal_topo_vect(sub_id, topo) 
                act = self.action_space({"set_bus":{"substations_id":[(sub_id, topo_legal)]}})
                sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)

                if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                    if not sim_done:
                        topo_dqn_act = act
                        act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                        if act_rho <= present_rho_threshold:
                            sim_rho.append(act_rho)
                            all_actions.append(act)
            
                # dqn topology action combined with the topology
                act = self.combine_with_redispatch(self, observation, topo_dqn_act)
                sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)

                if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                    if not sim_done:
                        if np.max(sim_obs.rho) <= present_rho_threshold:
                            sim_rho.append(np.max(sim_obs.rho))
                            all_actions.append(act)
                
                # combining dqn action with reconnection and redispatch action    
                if reconnect_act is not None:
                    act = self.combine_with_topology(reconnect_act, sub_id,  topo_legal, 1)
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    reco_topo_act = act
                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        if not sim_done:
                            if act_rho <= present_rho_threshold:
                                sim_rho.append(act_rho)
                                all_actions.append(act)

                    # topology and reconnection action combined with the topology action        
                    act = self.combine_with_redispatch(self, observation, reco_topo_act)
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        if not sim_done:
                            if act_rho <= present_rho_threshold:
                                sim_rho.append(act_rho)
                                all_actions.append(act)
                        
                # combining dqn action with disconnection and redispatch action
                     
                if disconnect_act is not None:
                    act = self.combine_with_topology(disconnect_act, sub_id,topo_legal, -1)
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)    
                    disco_topo_act = act
                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        if not sim_done:
                            if act_rho <= present_rho_threshold:
                                sim_rho.append(act_rho)
                                all_actions.append(act)
                    
                    act = self.combine_with_redispatch(self, observation, disco_topo_act)
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)                
                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        if not sim_done:
                            if act_rho <= present_rho_threshold:
                                sim_rho.append(act_rho)
                                all_actions.append(act)
        
            # heuristic actions 

            if self.observation.time_before_cooldown_sub[16] == 0:
                act = self.sub_16_act(observation,self.topo_action_sub_16 ,present_rho_threshold,do_nothing_reward)
                
                if act is not None:
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    sub_id_ = 16
                    topo_vector = act.to_vect()
                    if not sim_done:
                        sim_rho.append(act_rho)
                        all_actions.append(act)      


                    act = self.combine_with_redispatch(observation, act)
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                    if not sim_done:
                        if act_rho <= present_rho_threshold:
                            sim_rho.append(act_rho)
                            all_actions.append(act)
                    


                    

                    if reconnect_action is not None:
                        if (self.observation.line_or_to_subid[reconnect_act] != 16 or self.observation.line_ex_to_subid[reconnect_act] != 16):
                            
                            act_ = self.legal_line_topo_act(observation,topo_vector,reconnect_act)
                            act = self.action_space.from_vect(act_ + reconnect_action.to_vect())
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                            reco_topo_act = act
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:

                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)
                                        
                                    
                            act = self.combine_with_redispatch(observation, reco_topo_act)
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)
                                
                                    
                    if disconnect_act is not None:
                        if  self.observation.line_or_to_subid[disconnect_act] != 16 or self.observation.line_ex_to_subid[disconnect_act] != 16:
                            act_  = self.legal_line_topo_act(observation,topo_vector,disconnect_act)
                            
                            act = self.action_space.from_vect(act_ + disconnect_action.to_vect())
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)    
                            disco_topo_act = act
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])            
                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)
                                    
                            act = self.combine_with_redispatch(observation, disco_topo_act)
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)

            if self.observation.time_before_cooldown_sub[28] == 0:
                act = self.sub_28_act(observation,self.topo_action_sub_28 ,present_rho_threshold,do_nothing_reward)
                
                if act is not None:
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    sub_id_ = 28
                    topo_vector = act.to_vect()
                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])
                    if act_rho <= present_rho_threshold:
                        sim_rho.append(act_rho)
                        all_actions.append(act)

                    act = self.combine_with_redispatch(observation, act)
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                    if not sim_done:
                        if act_rho <= present_rho_threshold:
                            sim_rho.append(act_rho)
                            all_actions.append(act)


                    if reconnect_action is not None:
                        if (self.observation.line_or_to_subid[reconnect_act] != 28 or self.observation.line_ex_to_subid[reconnect_act] != 28):
                            
                            act_ = self.legal_line_topo_act(observation,topo_vector,reconnect_act)
                            act = self.action_space.from_vect(act_ + reconnect_action.to_vect())
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                            reco_topo_act = act
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:

                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)
                                        
                                    
                            act = self.combine_with_redispatch(observation, reco_topo_act)
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])  

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)
                                
                                    
                    if disconnect_act is not None:
                        if  self.observation.line_or_to_subid[disconnect_act] != 28 or self.observation.line_ex_to_subid[disconnect_act] != 28:
                            act_  = self.legal_line_topo_act(observation,topo_vector,disconnect_act)
                            
                            act = self.action_space.from_vect(act_ + disconnect_action.to_vect())
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)    
                            disco_topo_act = act
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])            
                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)
                            
                            act = self.combine_with_redispatch(observation, disco_topo_act)
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)

            if self.observation.time_before_cooldown_sub[23] == 0:

                act = self.sub_23_act(observation,self.topo_action_sub_23 ,present_rho_threshold,do_nothing_reward)
                
                if act is not None:
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    sub_id_ = 23
                    topo_vector = act.to_vect()

                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])
                    if act_rho <= present_rho_threshold:
                        sim_rho.append(act_rho)
                        all_actions.append(act)


                    act = self.combine_with_redispatch(observation, act)
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                    if not sim_done:
                        if act_rho <= present_rho_threshold:
                            sim_rho.append(act_rho)
                            all_actions.append(act)


                    if reconnect_action is not None:
                        if (self.observation.line_or_to_subid[reconnect_act] != 23 or self.observation.line_ex_to_subid[reconnect_act] != 23):
                            
                            act_ = self.legal_line_topo_act(observation,topo_vector,reconnect_act)
                            act = self.action_space.from_vect(act_ + reconnect_action.to_vect())
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                            reco_topo_act = act
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:

                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)
                                        
                                    
                            act = self.combine_with_redispatch(observation, reco_topo_act)
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)
                                
                                    
                    if disconnect_act is not None:
                        if  self.observation.line_or_to_subid[disconnect_act] != 23 or self.observation.line_ex_to_subid[disconnect_act] != 23:
                            act_  = self.legal_line_topo_act(observation,topo_vector,disconnect_act)
                            
                            act = self.action_space.from_vect(act_ + disconnect_action.to_vect())
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)    
                            disco_topo_act = act
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])            
                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)
                                    
                            act = self.combine_with_redispatch(observation, disco_topo_act)
                            sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                            act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if act_rho <= present_rho_threshold:
                                        sim_rho.append(act_rho)
                                        all_actions.append(act)

            if len(sim_rho) > 0:
                best_act_idx_rho = np.argmin(sim_rho)
                act_rho = all_actions[best_act_idx_rho]
            else:
                act_rho = do_nothing_action
            

            #why we are simulating again? 
            sim_obs_reward,sim_reward_,sim_done,sim_info = observation.simulate(act_rho)

            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                if not sim_done:
                    act = act_rho
            else:
                act = do_nothing_action
            

            
        else:
            if len(non_ol_acts) > 0:
                act_idx = np.argmax(non_ol_reward)
                act = non_ol_acts[act_idx]
               
            else:
                act = do_nothing_action

        return act

#heuristic actions for substation 23, 28 and 16 

    def sub_23_act(self,observation,sub_23_action,present_rho_threshold,do_nothing_reward):
        best_action = None
        sim_rewards_sub23 = []
        sim_rho_sub23 = []
        legal_sub_23_vect = []
        new_rho = None
        best_action_reward, rho_act = None, None
        substation_id_ = 23

        for i in range(len(sub_23_action)):
            topo_legal =  self.legal_topo_vect(substation_id_,sub_23_action[i][1] )
            action = self.action_space({"set_bus":{"substations_id":[(substation_id_, topo_legal)]}})
            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(action)


            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                if not sim_done:
                    sim_rewards_sub23.append(sim_reward)
                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])  
                    if act_rho <= present_rho_threshold:
                        sim_rho_sub23.append(act_rho)
                        legal_sub_23_vect.append(topo_legal)


        if len(legal_sub_23_vect) > 0:
            act_rho_id = np.argmin(sim_rho_sub23)
            action_ = legal_sub_23_vect[act_rho_id]
            best_action  = self.action_space({"set_bus":{"substations_id":[(substation_id_, action_)]}})
        else:
            best_action = None
        
        return best_action 

    def sub_28_act(self,observation,sub_28_action,present_rho_threshold,do_nothing_reward):
        best_action = None
        sim_rewards_sub28 = []
        sim_rho_sub28 = []
        legal_sub_28_vect = []
        new_rho = None
        best_action_reward, rho_act = None, None
        substation_id_ = 28

        for i in range(len(sub_28_action)):
            topo_legal = self.legal_topo_vect(substation_id_,sub_28_action[i][1] ) #[i][1]  
            action = self.action_space({"set_bus":{"substations_id":[(substation_id_, topo_legal)]}})
            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(action)

            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                if not sim_done:
                    sim_rewards_sub28.append(sim_reward)
                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])  

                    if act_rho <= present_rho_threshold:
                        sim_rho_sub28.append(act_rho)
                        legal_sub_28_vect.append(topo_legal)


        if len(legal_sub_28_vect) > 0:
            act_rho_id = np.argmin(sim_rho_sub28)
            action_ = legal_sub_28_vect[act_rho_id]
            best_action  = self.action_space({"set_bus":{"substations_id":[(substation_id_, action_)]}})
        else:
            best_action = None

        return best_action

    def sub_16_act(self,observation,sub_16_action,present_rho_threshold,do_nothing_reward):
        best_action = None
        sim_rewards_sub16 = []
        sim_rho_sub16 = []
        legal_sub_16_vect = []
        new_rho = None
        best_action_reward, rho_act = None, None
        substation_id_ = 16


        for i in range(len(sub_16_action)):
            topo_legal = self.legal_topo_vect(substation_id_,sub_16_action[i] )
            action = self.action_space({"set_bus":{"substations_id":[(substation_id_, topo_legal)]}})
            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(action)

            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                if not sim_done:
                    sim_rewards_sub16.append(sim_reward)
                    act_rho = np.sum(sorted(sim_obs.rho, reverse=True)[:2])  

                    if act_rho <= present_rho_threshold:
                        sim_rho_sub16.append(act_rho)
                        legal_sub_16_vect.append(topo_legal)


        if len(legal_sub_16_vect) > 0:
            act_rho_id = np.argmin(sim_rho_sub16)
            action_ = legal_sub_16_vect[act_rho_id]
            best_action  = self.action_space({"set_bus":{"substations_id":[(substation_id_, action_)]}})
        else:
            best_action = None

        return best_action
    
    def combine_with_redispatch(self, observation, action):
        do_nothing_action = self.action_space({})

        if (observation.line_status[45] == False or observation.line_status[56] == False)  and action != do_nothing_action \
                and self.redispatch_cnt < self.max_redispatch_cnt \
                and action.impact_on_objects()['topology']['changed']:
            sim_obs, simu_reward, simu_done, sim_info = observation.simulate(action)
            assert not sim_info['is_illegal'] and not sim_info['is_ambiguous']
        
            origin_rho = 10.0
            if not simu_done:
                origin_rho = sim_obs.rho.max()

            least_rho = origin_rho
            best_action = None

            for redispatch_action in self.redispatch_actions:
                combine_action = self.action_space.from_vect(action.to_vect() + redispatch_action.to_vect())   
                sim_obs, simu_reward, simu_done, sim_info = observation.simulate( combine_action)

                assert not sim_info['is_illegal'] and not sim_info['is_ambiguous']

                max_rho = 10.0
                if not simu_done:
                    max_rho = sim_obs.rho.max()
                if max_rho < least_rho:
                    least_rho = max_rho
                    best_action = combine_action

            if least_rho < origin_rho:
                action = best_action
                self.redispatch_cnt += 1 # max 3 times # recovery 45/56 

        return action
    #adding redispatch updated code with 40 actions to control the redispacth amount
    
    def _reset_redispatch(self):
        if np.max(self.observation.rho) < 1.0:
            # reset redispatch
            if not np.all(self.observation.target_dispatch == 0.0):
                gen_ids = np.where(self.observation.gen_redispatchable)[0]
                gen_ramp = self.observation.gen_max_ramp_up[gen_ids]
                changed_idx = np.where(self.observation.target_dispatch[gen_ids] != 0.0)[0]
                redispatchs = []
                for idx in changed_idx:
                    target_value = self.observation.target_dispatch[gen_ids][idx]
                    value = min(abs(target_value), gen_ramp[idx])
                    value = -1 * target_value / abs(target_value) * value
                    redispatchs.append((gen_ids[idx], value))

                act = self.action_space({"redispatch": redispatchs})
                obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(act)

                assert not info_simulate['is_illegal'] and not info_simulate['is_ambiguous']

                if not done_simulate and obs_simulate is not None and not any(np.isnan(obs_simulate.rho)):
                    if np.max(obs_simulate.rho) < 1.0:
                        return act
    
    def getAction(self, observation, epsilon,train=True):    # modification in act function before test

        # Fallback Action - Do Nothing
        action_for_buffer = {}
        do_nothing_action = self.action_space({})
        act = do_nothing_action

        rew_counter  = -1
        rho_counter  = -1

        all_rho_acts = []
        all_rew_acts = []
        non_ol_acts  = []

        non_ol_rho = []
        sim_rewards   = []
        sim_rho       = []

        self.observation = observation

        disconnect_act  = None
        reconnect_act   = None
        recovery_action = None
        redispatch_act  = None
        recovery_redisp_act = None
        action_for_buffer = {}
        rew_action_flag = {}
        rho_action_flag = {}
        topo_act_2_buff = {}

        features = []
        dqn_topo_buffer   = None
        redispatch_buffer = None
        ddpg_topo_buffer = None
        # step 1 : do nothing action as a fallback action

        do_nothing_action = self.action_space({})
        sim_obs,sim_reward,sim_done,_ = observation.simulate(do_nothing_action)
        do_nothing_reward = sim_reward

        present_rho = np.max(sim_obs.rho)

        sim_rewards.append(sim_reward)
        all_rew_acts.append(do_nothing_action)
        rew_counter += 1
        rew_action_flag[rew_counter] = 'do_nothing'

        non_ol_acts.append(do_nothing_action)
        non_ol_rho.append(sim_reward)

        if np.max(sim_obs.rho) <= present_rho:
            sim_rho.append(np.max(sim_obs.rho))
            all_rho_acts.append(do_nothing_action)
            rho_counter += 1
            rho_action_flag[rho_counter] = 'do_nothing'

        # step 2 Reconnect Action
        act, line_id = self._reconnect_action()
        reconnect_act = line_id

        if act is not None:
            sim_obs,sim_reward,sim_done,_ = observation.simulate(act)

            if sim_reward > do_nothing_reward:
                sim_rewards.append(sim_reward)
                all_rew_acts.append(act)
                rew_counter += 1
                rew_action_flag[rew_counter] = 'reconnect'

            if np.max(sim_obs.rho) <= present_rho:
                sim_rho.append(np.max(sim_obs.rho))
                all_rho_acts.append(act)
                rho_counter += 1
                rho_action_flag[rho_counter] = 'reconnect'

        #step 3 check if recover actions are present 

        if self.observation is not None and not any(np.isnan(self.observation.rho)):
            if np.all(self.observation.topo_vect != -1):

                act = self._reset_redispatch()
                recovery_redisp_act = act
                if recovery_redisp_act is not None:
                    sim_obs,sim_reward,sim_done,_ = observation.simulate(act)

                    if sim_reward > do_nothing_reward:
                        sim_rewards.append(sim_reward)
                        all_rew_acts.append(act)
                        rew_counter += 1
                        rew_action_flag[rew_counter] = 'recovery_redispatch'

                    if np.max(sim_obs.rho) <= present_rho:
                        sim_rho.append(np.max(sim_obs.rho))
                        all_rho_acts.append(act)
                        rho_counter += 1
                        rho_action_flag[rho_counter] = 'recovery_redispatch'


                act = self._reset_topology()
                recovery_action = act
                if recovery_action is not None:
                    sim_obs,sim_reward,sim_done,_ = observation.simulate(act)

                    if sim_reward > do_nothing_reward:
                        sim_rewards.append(sim_reward)
                        all_rew_acts.append(act)
                        rew_counter += 1
                        rew_action_flag[rew_counter] = 'recovery'

                    if np.max(sim_obs.rho) <= present_rho:
                        sim_rho.append(np.max(sim_obs.rho))
                        all_rho_acts.append(act)
                        rho_counter += 1
                        rho_action_flag[rho_counter] = 'recovery'



        ol_list = self.getRankedOverloadList(observation)

        if len(ol_list):
            # step 4 Line Disconnection Action
            line_id = self.lineModel.select_Action(observation, self.num_lines,
                                                   self.gen_pmax, epsilon, train)

            if line_id is not None:
                act = self.action_space({"set_line_status": [(line_id,-1)]})
                disconnect_act = line_id
                sim_obs,sim_reward,sim_done,_ = observation.simulate(act)

                if sim_reward > do_nothing_reward:
                    sim_rewards.append(sim_reward)
                    all_rew_acts.append(act)
                    rew_counter += 1
                    rew_action_flag[rew_counter] = 'disconnect'

                if np.max(sim_obs.rho) <= present_rho:
                    sim_rho.append(np.max(sim_obs.rho))
                    all_rho_acts.append(act)
                    rho_counter += 1
                    rho_action_flag[rho_counter] = 'disconnect'

            # step 5 ddpg topological action  and combined with line actions

            state = self.utils.convert_obs(observation)
            for indx, sub_id in enumerate(self.sub_id):
                noise = self.noise[indx].noise()
                
                policy = self.topoModel_DDPG[indx].actor_act(state, self.kdtree[indx].comb_len, noise)                
                nn_idx = self.kdtree[indx].get_nn_topos(policy, self.nn)
                nn_actions = []

                for idx in nn_idx:
                    nn_actions.append(self.kdtree[indx].get_actions(idx))   

                nn_actions = np.array(nn_actions, dtype='int')
                enc_topos  = self.encoder[indx].encode(nn_actions).detach().cpu().numpy()
                enc_topos  = enc_topos.reshape(self.nn, self.action_size)                
                q_vals = self.topoModel_DDPG[indx].critic_act(state, enc_topos)                
                q_vals = q_vals.detach().cpu().numpy().flatten()                
                sorted_idx = np.argsort(q_vals)[::-1]
                q_indx = np.argmax(q_vals)                  # index of max_qval
                tp_idx = nn_idx[q_indx]                     # get actual topo index 

                # For topological action alone
                for q_indx in sorted_idx:             
                    tp_idx = nn_idx[q_indx]                    
                    
                    if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                        topo_ = self.topo_combos[indx][tp_idx]                        
                        topo_legal = self.legal_topo_vect(sub_id, topo_)
                        act = self.action_space({"set_bus":{"substations_id": [(sub_id, topo_legal)]}})
                        
                        sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
                        act_rho = np.max(sim_obs.rho) #np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                        if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                            if not sim_done:
                                if np.max(sim_obs.rho) <= present_rho:
                                    topo_act_2_buff[sub_id] = [enc_topos[q_indx], indx]
                                    sim_rho.append(np.max(sim_obs.rho))
                                    all_rho_acts.append(act)
                                    rho_counter += 1
                                    rho_action_flag[rho_counter] = 'topo_ddpg_' + str(sub_id)
                                    break 


                # combining the ddpg action and redispatch
                for q_indx in sorted_idx:             
                    tp_idx = nn_idx[q_indx]                    
                    
                    if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                        topo_ = self.topo_combos[indx][tp_idx]                        
                        topo_legal = self.legal_topo_vect(sub_id, topo_)
                        act = self.action_space({"set_bus":{"substations_id": [(sub_id, topo_legal)]}})
                        act = self.combine_with_redispatch(observation, act)
                        sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
                        act_rho = np.max(sim_obs.rho) #np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                        if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                            if not sim_done:
                                if np.max(sim_obs.rho) <= present_rho:
                                    sim_rho.append(np.max(sim_obs.rho))
                                    all_rho_acts.append(act)
                                    rho_counter += 1
                                    rho_action_flag[rho_counter] = 'topo_ddpg_redispatch' + str(sub_id)
                                    break 


                # If reconnect action is present, combining it with the ddpg substation 16 action
                if reconnect_act is not None:                               
                    for q_indx in sorted_idx:             
                        tp_idx = nn_idx[q_indx]

                        if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                            topo_ = self.topo_combos[indx][tp_idx]                        
                            topo_legal = self.legal_topo_vect(sub_id, topo_)
                            act = self.combine_with_topology(reconnect_act, sub_id, topo_legal, 1)
                            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
                            act_rho = np.max(sim_obs.rho) #np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if np.max(sim_obs.rho) <= present_rho:
                                        sim_rho.append(np.max(sim_obs.rho))
                                        all_rho_acts.append(act)
                                        rho_counter += 1
                                        rho_action_flag[rho_counter] = 'reconnect_topo_ddpg_'+str(sub_id)
                                        break 

                    for q_indx in sorted_idx:             
                        tp_idx = nn_idx[q_indx]

                        if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                            topo_ = self.topo_combos[indx][tp_idx]                        
                            topo_legal = self.legal_topo_vect(sub_id, topo_)
                            act = self.combine_with_topology(reconnect_act, sub_id, topo_legal, 1)
                            
                            act = self.combine_with_redispatch(observation, act)
                            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
                            act_rho = np.max(sim_obs.rho) #np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if np.max(sim_obs.rho) <= present_rho:
                                        sim_rho.append(np.max(sim_obs.rho))
                                        all_rho_acts.append(act)
                                        rho_counter += 1
                                        rho_action_flag[rho_counter] = 'reconnect_redispatch_topo_ddpg_'+str(sub_id)
                                        break 


                
                
                # If disconnect action is present, combining it with the disconnect action
                if disconnect_act is not None:                               
                    for q_indx in sorted_idx:             
                        tp_idx = nn_idx[q_indx]
                        
                        if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                            topo_ = self.topo_combos[indx][tp_idx]                        
                            topo_legal = self.legal_topo_vect(sub_id, topo_)
                            act = self.combine_with_topology(disconnect_act, sub_id, topo_legal, -1)
                            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

                            act_rho = np.max(sim_obs.rho) #np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if np.max(sim_obs.rho) <= present_rho:
                                        sim_rho.append(np.max(sim_obs.rho))
                                        all_rho_acts.append(act)
                                        rho_counter += 1
                                        rho_action_flag[rho_counter] = 'disconnect_topo_ddpg_'+str(sub_id)
                                        break 

                    for q_indx in sorted_idx:             
                        tp_idx = nn_idx[q_indx]
                        
                        if self.observation.time_before_cooldown_sub[sub_id] == 0:                    
                            topo_ = self.topo_combos[indx][tp_idx]                        
                            topo_legal = self.legal_topo_vect(sub_id, topo_)
                            act = self.combine_with_topology(disconnect_act, sub_id, topo_legal, -1)
                            act = self.combine_with_redispatch(observation, act)

                            sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)
                            act_rho = np.max(sim_obs.rho) #np.sum(sorted(sim_obs.rho, reverse=True)[:2])

                            if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                                if not sim_done:
                                    if np.max(sim_obs.rho) <= present_rho:
                                        sim_rho.append(np.max(sim_obs.rho))
                                        all_rho_acts.append(act)
                                        rho_counter += 1
                                        rho_action_flag[rho_counter] = 'disconnect_redispatch_topo_ddpg_'+ str(sub_id)
                                        break 

            # step 6 Topo DQN
            if train:
                if np.random.random() <= epsilon:
                    act, dqn_topo_buffer, topo, sub_id = self.exploring_actions(observation)
                else:
                    act, dqn_topo_buffer, topo, sub_id = self.topoModel.select_action(state,
                                                                                      observation,
                                                                                      self.action_space)
            else:
                act, dqn_topo_buffer, topo, sub_id = self.topoModel.select_action(state,
                                                                                  observation,
                                                                                  self.action_space)

            if (act != None and self.observation.time_before_cooldown_sub[sub_id] == 0):

                topo_legal = self.legal_topo_vect(sub_id, topo)
                act = self.action_space({"set_bus":{"substations_id":[(sub_id, topo_legal)]}})

                sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

                if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                    if not sim_done:

                        topo_dqn_act = act
                        sim_rewards.append(sim_reward)
                        all_rew_acts.append(act)
                        rew_counter += 1
                        rew_action_flag[rew_counter] = 'topo_dqn'

                        if np.max(sim_obs.rho) <= present_rho:
                            sim_rho.append(np.max(sim_obs.rho))
                            all_rho_acts.append(act)
                            rho_counter += 1
                            rho_action_flag[rho_counter] = 'topo_dqn'

                # combining the dqn action with the line action, adding redispatch action 

                act = self.combine_with_redispatch(self, observation, topo_dqn_act)
                sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)

                if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                    if not sim_done:

                        if sim_reward > do_nothing_reward:
                            sim_rewards.append(sim_reward)
                            all_rew_acts.append(act)
                            rew_counter += 1
                            rew_action_flag[rew_counter] = 'redispatch_topo_dqn'

                        if np.max(sim_obs.rho) <= present_rho:
                            sim_rho.append(np.max(sim_obs.rho))
                            all_rho_acts.append(act)
                            rho_counter += 1
                            rho_action_flag[rho_counter] = 'redispatch_topo_dqn'

#combining dqn with the reconnect and later both actions with the redispatch action
                if reconnect_act is not None:
                    act = self.combine_with_topology(reconnect_act, sub_id,
                                                     topo_legal, 1)
                    sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        if not sim_done:

                            if sim_reward > do_nothing_reward:
                                sim_rewards.append(sim_reward)
                                all_rew_acts.append(act)
                                rew_counter += 1
                                rew_action_flag[rew_counter] = 'reconnect_topo_dqn'

                            if np.max(sim_obs.rho) <= present_rho:
                                sim_rho.append(np.max(sim_obs.rho))
                                all_rho_acts.append(act)
                                rho_counter += 1
                                rho_action_flag[rho_counter] = 'reconnect_topo_dqn'

                    act = self.combine_with_redispatch(self, observation, act)
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)

                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        if not sim_done:
                            if sim_reward > do_nothing_reward:
                                sim_rewards.append(sim_reward)
                                all_rew_acts.append(act)
                                rew_counter += 1
                                rew_action_flag[rew_counter] = 'reconnect_topo_dqn_redispatch'

                            if np.max(sim_obs.rho) <= present_rho:
                                sim_rho.append(np.max(sim_obs.rho))
                                all_rho_acts.append(act)
                                rho_counter += 1
                                rho_action_flag[rho_counter] = 'reconnect_topo_dqn_redispatch'


                if disconnect_act is not None:
                    act = self.combine_with_topology(disconnect_act, sub_id,
                                                     topo_legal, -1)
                    sim_obs,sim_reward,sim_done,sim_info = observation.simulate(act)

                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        if not sim_done:

                            if sim_reward > do_nothing_reward:
                                sim_rewards.append(sim_reward)
                                all_rew_acts.append(act)
                                rew_counter += 1
                                rew_action_flag[rew_counter] = 'disconnect_topo_dqn'

                            if np.max(sim_obs.rho) <= present_rho:
                                sim_rho.append(np.max(sim_obs.rho))
                                all_rho_acts.append(act)
                                rho_counter += 1
                                rho_action_flag[rho_counter] = 'disconnect_topo_dqn'

                    act = self.combine_with_redispatch(self, observation, act)
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)

                    if not sim_info["is_illegal"] and not sim_info['is_ambiguous']:
                        if not sim_done:

                            if sim_reward > do_nothing_reward:
                                sim_rewards.append(sim_reward)
                                all_rew_acts.append(act)
                                rew_counter += 1
                                rew_action_flag[rew_counter] = 'disconnect_topo_dqn_redispatch'

                            if np.max(sim_obs.rho) <= present_rho:
                                sim_rho.append(np.max(sim_obs.rho))
                                all_rho_acts.append(act)
                                rho_counter += 1
                                rho_action_flag[rho_counter] = 'disconnect_topo_dqn_redispatch'

            # Explore DQN for sub 16 (sub ids with > threshold topo combos)

            sub_id_ = 16
            if observation.time_before_cooldown_sub[sub_id_] == 0:
                act = self.sub_16_act(observation, self.topo_action_sub_16,
                                      present_rho, do_nothing_reward)

                if act is not None:
                    sim_obs,sim_reward,sim_done,sim_info  = observation.simulate(act)
                    topo_vector = act.to_vect()

                    sim_rewards.append(sim_reward)
                    all_rew_acts.append(act)
                    rew_counter += 1
                    rew_action_flag[rew_counter] = 'topo_sub16'


                    sim_rho.append(sim_reward)
                    all_rho_acts.append(act)
                    rho_counter += 1
                    rho_action_flag[rho_counter] = 'topo_sub16'

            # Explore Topo, Redisp, Line Combos

            if len(sim_rho) == 0:
                act = do_nothing_action
    
            elif len(sim_rho) > 0:
                best_act_idx_rho = np.argmin(sim_rho)
                act_rho = all_rho_acts[best_act_idx_rho]
                sim_obs_rho,sim_rho_reward,sim_done,sim_info = observation.simulate(act_rho)

                act = act_rho
                best_act_idx = best_act_idx_rho
                action_flag = rho_action_flag

                if 'disconnect' in action_flag[best_act_idx]:
                    action_for_buffer['disconnect'] = disconnect_act
                if 'reconnect' in action_flag[best_act_idx]:
                    action_for_buffer['reconnect']  = None
                if 'topo_dqn' in action_flag[best_act_idx]:
                    action_for_buffer['topo_dqn']   = dqn_topo_buffer
                if 'topo_sub16' in action_flag[best_act_idx]:
                    action_for_buffer['topo_sub16'] = None
                if 'do_nothing' in action_flag[best_act_idx]:
                    action_for_buffer['do_nothing'] = None
                if 'recovery' in action_flag[best_act_idx]:
                    action_for_buffer['recovery'] = None
                if 'recovery_redispatch' in action_flag[best_act_idx]:
                    action_for_buffer['recovery_redispatch'] = None

                elif 'topo_ddpg' in action_flag[best_act_idx]:
                    sub_id = int(action_flag[best_act_idx].split('_')[-1])
                    action_for_buffer['topo_ddpg'] = topo_act_2_buff[sub_id]


            return act, action_for_buffer, features

    def get_available_dispatch_actions(self,observations):
        action_array = []
        max_up = observations.gen_max_ramp_up
        max_down = observations.gen_max_ramp_down
        for i in range(len(max_up)):
            if max_up[i] != 0:
                action = self.action_space({"redispatch": [(i , +max_up[i])]})
                action_array.append(action)
                action = self.action_space({"redispatch": [(i , +max_up[i]/2)]})
                action_array.append(action)
                action = self.action_space({"redispatch": [(i , -max_down[i]/2)]})
                action_array.append(action)
                action = self.action_space({"redispatch":[(i , -max_down[i])]})
                action_array.append(action)

        return action_array

    def _reset_topology(self):
        if np.max(self.observation.rho) <= 0.95:  #check if max rho less than 0.95
            for sub_id, sub_elem_num in enumerate(self.observation.sub_info): #enumate subid and sub element
                sub_topo = self.observation.state_of(substation_id=sub_id)["topo_vect"]   #self.sub_toop_dict[sub_id] #getting topology of given sub id from dict

                if sub_id == 28:
                    sub28_topo = np.array([2, 1, 2, 1, 1]) #
                    if not np.all(
                            sub_topo.astype(int) == sub28_topo.astype(int)
                    ) and self.observation.time_before_cooldown_sub[
                            sub_id] == 0:
                        sub_id = 28

                        act = self.action_space({ "set_bus": {"substations_id": [(sub_id, sub28_topo)]}})

                        obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(act)
                        assert not info_simulate[
                            'is_illegal'] and not info_simulate['is_ambiguous']
                        if not done_simulate and obs_simulate is not None and not any(
                                np.isnan(obs_simulate.rho)):
                            if np.max(obs_simulate.rho) <= 1:
                                return act
                    continue



                if np.any(sub_topo == 2) and self.observation.time_before_cooldown_sub[sub_id] == 0:
                    sub_topo = np.where(sub_topo == 2, 1,sub_topo)  # bus 2 to bus 1
                    sub_topo = np.where(sub_topo == -1, 0, sub_topo)  # don't do action in bus=-1
                    reconfig_sub = self.action_space({ "set_bus": { "substations_id": [(sub_id, sub_topo)] } })

                    obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(reconfig_sub)
                    # self.observation._obs_env._reset_to_orig_state()

                    assert not info_simulate['is_illegal'] and not info_simulate['is_ambiguous']

                    if not done_simulate:
                        assert np.any( obs_simulate.topo_vect !=  self.observation.topo_vect)  # have some impact

                    if not done_simulate and obs_simulate is not None and not any( np.isnan(obs_simulate.rho)):
                        if np.max(obs_simulate.rho) <= 1:
                            return reconfig_sub


        if np.max(self.observation.rho) >= 1.0:
            sub_id = 28
            sub_topo =  self.observation.state_of(substation_id=sub_id)["topo_vect"] # self.sub_toop_dict[sub_id]
            if np.any(sub_topo == 2 ) and self.observation.time_before_cooldown_sub[sub_id] == 0:
                sub28_topo = np.array([1, 1, 1, 1, 1])
                act = self.action_space({ "set_bus": {"substations_id": [(sub_id, sub28_topo)] }})
                obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate( act)
                assert not info_simulate['is_illegal'] and not info_simulate['is_ambiguous']
                if not done_simulate and obs_simulate is not None and not any(np.isnan(obs_simulate.rho)):
                    if np.max(obs_simulate.rho) <= 0.99:
                        return act


    def Q_vals_action_encoding(self,obs, Q, a_space):
        sub_id = []

        actions_topology = []

        acts = []

        a = self.df[Q][0] #substation id

        arr = copy.deepcopy(self.sub_topo[a])
        arr_2 = copy.deepcopy(self.sub_topo[a])

        for node in self.df[Q][1][0]:
            arr[node-1] = 1
            arr_2[node-1] = 2
        for node in self.df[Q][1][1]:
            arr[node-1] = 2
            arr_2[node-1] = 1

        acts.append(arr)
        acts.append(arr_2)
        actions_topology.append((a,acts))
        sub_id.append(a)

        sub_act = []
        sub_ID = []

        for j in range(len(actions_topology)):
            sub_id = actions_topology[j][0]
            # if sub_id != 64:
            for l in range(len(actions_topology[j][1])):
                act = a_space({"set_bus": {"substations_id": [(sub_id, actions_topology[j][1][l])]}})
                sub_act.append(act)
                sub_ID.append([sub_id, actions_topology[j][1][l]])

            # else:
            #     sub_act.append(a_space({}))

        return actions_topology,sub_act


    def combine_with_topology(self, line_id, sub_id, topo_act, line_act):
        new_line_status_array = np.zeros(self.observation.rho.shape).astype(int)
        new_line_status_array[line_id] = line_act

        line_el_idx = -1
        # check if line is in substation
        # If the sub is the line origin
        if sub_id == self.observation.line_or_to_subid[line_id]:
            line_el_idx = self.observation.line_or_to_sub_pos[line_id]

        # If the sub is the line extrimity
        elif sub_id == self.observation.line_ex_to_subid[line_id]:
            line_el_idx = self.observation.line_ex_to_sub_pos[line_id]

        if line_el_idx > -1:
            topo_act[line_el_idx] = 0

        action_space_ = {}
        action_space_["set_bus"] = {}

        action_space_["set_bus"]["substations_id"] =  [(sub_id, topo_act)]
        action_space_["set_line_status"] = new_line_status_array

        act_combined = self.action_space(action_space_)

        return act_combined


    def legal_topo_vect(self,sub_id, topo_vect_):

        topo_vect = self.observation.state_of(substation_id=sub_id)["topo_vect"]

        do_nothing_indices = np.where(topo_vect < 0)[0]
        if len(do_nothing_indices)>0:
            for i in do_nothing_indices:
                topo_vect_[i] = 0
        return topo_vect_


    def _reconnect_action(self):
        disconnected = np.where(self.observation.line_status == False)[0].tolist()
        action = None
        line_id = 0
        line_ids = []
        sim_rhos_ = []
        for line_id in disconnected:
            if self.observation.time_before_cooldown_line[line_id] == 0:
                action = self.action_space({"set_line_status": [(line_id, +1)]})
                obs_simulate, reward_simulate, done_simulate, info_simulate = self.observation.simulate(action)

                sim_rhos_.append(np.max(obs_simulate.rho))
                line_ids.append(line_id)


            if len(line_ids) > 0:
                action_id_ = np.argmin(sim_rhos_)
                line_act_ = line_ids[action_id_]

                action = self.action_space({"set_line_status": [(line_act_, +1)]})
                # self.observation._obs_env._reset_to_orig_state()

                # if np.max(self.observation.rho) <= 0.95 and np.max(obs_simulate.rho) >= 0.95:
                #     continue

                # line_id = line_id
        return action,line_id

    def getRankedOverloadList(self, observation):
        sort_rho = -np.sort(-observation.rho)  # sort in descending order for positive values
        sort_indices = np.argsort(-observation.rho)
        line_list = [sort_indices[i] for i in range(len(sort_rho))
                     if sort_rho[i] >= self.rho_threshold]
        return line_list


    def exploring_actions(self, observation):
        counter  = 0
        act_flag = {}
        random_actions = np.random.choice(np.arange(self.dqn_action_space),
                                          self.dqn_action_space,
                                          replace=False)
        for indx, q_indx in enumerate(random_actions):
            topo_vect,enc_act = self.Q_vals_action_encoding(observation,q_indx, self.action_space)

            best_action,best_action_id = self.select_best_action_from_dual(observation, enc_act)
            topo_vect_ = topo_vect[0][1][best_action_id]
            substation_topo = topo_vect[0][0]

            _,_,sim_done,sim_info = observation.simulate(best_action)

            if not sim_info['is_illegal']:
                if not sim_done:

                    topo_act = best_action
                    # sim_rho.append(observation.simulate(topo_act)[1])
                    act_flag[counter] = 'topology'
                    counter += 1
                    act_id = q_indx
                    return topo_act, act_id,topo_vect_,substation_topo

        return None, None, None, None


    def select_best_action_from_dual(self, observation, encoded_action):
        best_action = None
        enc_act = encoded_action
        act_flag = {}
        counter  = 0
        a_reward = []

        for i in range(len(enc_act)):
            _,sim_reward,sim_done,sim_info = observation.simulate(enc_act[i])
            a_reward.append(sim_reward)
            act_flag[counter] = 'enc_act_' + str(i)
            counter +=1


        best_act_idx = np.argmax(a_reward)

        if act_flag[best_act_idx] == 'enc_act_0':
            best_action = enc_act[0]

        elif act_flag[best_act_idx] == 'enc_act_1':
            best_action = enc_act[1]

        # print('best action indexes ', best_act_idx)


        return best_action,best_act_idx

    def train_dqn(self, current_step_num, batch_size, discount, policy_freq):
        if batch_size <= self.replay_buffer_dqn.getLength():
            return self.topoModel.train(self.replay_buffer_dqn,
                                   current_step_num,
                                   batch_size, discount,
                                   policy_freq)

    def train_redispatch_dqn(self, current_step_num, batch_size, discount, policy_freq):
        if batch_size <= self.replay_buffer_dqn_redispatch.getLength():
            return self.redispatchModel.train(self.replay_buffer_dqn_redispatch,
                                    current_step_num,
                                    batch_size, discount,
                                    policy_freq)


    def train(self, current_step_num, batch_size, discount, policy_freq):
        # print(self.sub_id)
        for i in range(len(self.sub_id)):

            if batch_size <= self.replay_buffer_ddpg[i].getLength():
                # print(i, self.sub_id[i])

                self.topoModel_DDPG[i].train(current_step_num, batch_size,
                                     discount, policy_freq,
                                     self.replay_buffer_ddpg[i],
                                     self.kdtree[i],
                                     self.encoder[i])

    def update_PolicyNet(self):
        return self.lineModel.updatePolicy()

def make_agent(env, this_directory_path):
    # Add l2rpn reward
    my_agent = MyAgent(env, this_directory_path)
    my_agent.load(this_directory_path)
    return my_agent
