import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
import numpy as np
from multiprocessing import Pool
import sys

class QLearner(object):
    """ 
    A (rudimentary) general Q-Learner class that implements
    Q-learning on arbitrary state and action space.
    """

    def __init__(self, state_space, action_space, discount_factor):
        self.state_space = state_space
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.qtable = dict()

        # Q-Learner needs to span how many states and actions?
        total_num_states = 1
        for attribute in self.state_space.iterkeys():
            total_num_states *= len(self.state_space[attribute])
        #print "Q-Learner has {} possible states and {} actions".format(total_num_states, len(self.action_space))

    def optimal_action(self, state):
        q = []
        for action in self.action_space:
            qtable_key = self.tuplize(state, action)
            if qtable_key not in self.qtable:
                return random.choice(self.action_space)
            else:
                q.append(self.qtable[qtable_key])
        
        return self.action_space[q.index(max(q))]

    def update(self, state, action, reward, next_state, learning_rate):
        assert learning_rate >=0 and learning_rate <= 1, "Invalid learning rate"
        
        qtable_key = self.tuplize(state, action)
        if qtable_key not in self.qtable:
            self.qtable[qtable_key] = 0
        
        q_prime = []
        for act in self.action_space:
            qtable_key_tmp = self.tuplize(next_state, act)
            if qtable_key_tmp not in self.qtable:
                self.qtable[qtable_key_tmp] = 0
            q_prime.append(self.qtable[qtable_key_tmp])
        
        tmp = reward + self.discount_factor * max(q_prime)
        self.qtable[qtable_key] = learning_rate * tmp + (1 - learning_rate) * self.qtable[qtable_key]
        #print "Q-Learner Update:", qtable_key, self.qtable[qtable_key]

    def tuplize(self, state, action):
        return_value = [state[x] for x in sorted(state)]
        return_value.append(action)
        return tuple(return_value)

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    allowed_actions = [None, 'forward', 'left', 'right']
    state_space = {
        'next_waypoint': allowed_actions, 
        'oncoming': allowed_actions,
        'left': allowed_actions,
        'light': ['green', 'red']
    }

    def __init__(self, env, exploration_factor=0.8, discount_factor=0.8, learning_factor=1.0):        
        assert learning_factor > 0. and learning_factor < 500., "Invalid learning factor!"
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # Initialize additional variables here
        self.exploration_factor = exploration_factor
        self.qlearner = QLearner(self.state_space, self.allowed_actions, discount_factor)
        self.cumulative_reward = 0
        self.n_penalties = 0
        self.exploration_amount = 1.0 # Starts with 1.0        
        self.learning_factor = learning_factor

    def reset(self, destination=None):
        self.planner.route_to(destination)

        # Prepare for a new trip; reset any variables here, if required
        self.cumulative_reward = 0
        self.n_penalties = 0
        self.exploration_amount *= self.exploration_factor # Decays exponentially every iteration

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state before taking action
        self.state = {
            'next_waypoint': self.next_waypoint,
            'oncoming': inputs['oncoming'],
            'left': inputs['left'],
            'light': inputs['light']
        }
        
        # Select action according to policy that has decaying exploration
        if random.random() > self.exploration_amount:
            action = self.qlearner.optimal_action(self.state)
        else:
            action = random.choice(self.allowed_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Update state after taking action
        next_waypoint_after_action = self.planner.next_waypoint()
        inputs_after_action = self.env.sense(self)
        state_after_action = {
            'next_waypoint': next_waypoint_after_action,
            'oncoming': inputs_after_action['oncoming'],
            'left': inputs_after_action['left'],
            'light': inputs_after_action['light']
        }

        # Learn policy based on state, action, reward
        self.qlearner.update(self.state, action, reward, state_after_action, self.learning_factor/(self.learning_factor+t+1))
        
        # Update other variables for debug
        self.cumulative_reward += reward
        if reward < 0:
            self.n_penalties += 1

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run(epsilon = 0.8, gamma = 0.8, alpha_prime=1.0, update_delay=0.05, display=True):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, exploration_factor=epsilon, discount_factor=gamma, learning_factor=alpha_prime)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=update_delay, display=display)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    stats = sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    return stats

def unpack_run(arg):
    return run(*arg)

def analyze(num_sim_runs=100):
    """Run simulation multiple times with different parameters"""
    
    # Create a DataFrame to store all the analysis info
    analysis = pd.DataFrame(
        columns=[
            'exploration_rate',
            'discount_factor',
            'learning_factor',
            'mean_normed_time_left',
            'std_normed_time_left',
            'mean_normed_n_penalties',
            'std_normed_n_penalties',
            'mean_normed_cumulative_reward',
            'std_normed_cumulative_reward',
            'dest_reached_rate']
            )

    count = 0

    for epsilon in np.linspace(0.1, 1.0, 40):
        for gamma in np.linspace(0.1, 1.0, 40):
            for alpha_prime in np.linspace(1.0, 8.0, 3):
                stats = pd.DataFrame()

                epsilon_arr = np.empty(num_sim_runs)
                gamma_arr = np.empty(num_sim_runs)
                alpha_prime_arr = np.empty(num_sim_runs)
                update_delay_arr = np.zeros(num_sim_runs)
                display_arr = np.empty(num_sim_runs, dtype=bool)
                epsilon_arr.fill(epsilon)
                gamma_arr.fill(gamma)
                alpha_prime_arr.fill(alpha_prime)
                display_arr.fill(False)

                print "Measuring for epsilon {}, gamma {}, alpha_prime {}".format(epsilon, gamma, alpha_prime)

                pool = Pool()
                ret = pool.map(unpack_run, zip(epsilon_arr, gamma_arr, alpha_prime_arr, update_delay_arr, display_arr))
                pool.close()
                pool.join()

                for ret_val in ret:
                    stats = stats.append(ret_val)
                
                normed_time_left = stats['time_left'] / stats['deadline']
                normed_n_penalties = stats['n_penalties'] / stats['deadline']
                normed_cumulative_reward = stats['cumulative_reward'] / stats['deadline']
                dest_reached_rate = float(stats['dest_reached'].value_counts()[True]) / stats['dest_reached'].shape[0]

                mean_normed_time_left = np.mean(normed_time_left)
                std_normed_time_left = np.std(normed_time_left)
                mean_normed_n_penalties = np.mean(normed_n_penalties)
                std_normed_n_penalties = np.std(normed_n_penalties)
                mean_normed_cumulative_reward = np.mean(normed_cumulative_reward)
                std_normed_cumulative_reward = np.std(normed_cumulative_reward)

                count += 1

                analysis.loc[count] = (
                    epsilon, 
                    gamma,
                    alpha_prime, 
                    mean_normed_time_left,
                    std_normed_time_left, 
                    mean_normed_n_penalties,
                    std_normed_n_penalties,
                    mean_normed_cumulative_reward,
                    std_normed_cumulative_reward,
                    dest_reached_rate
                    )

    print "Ran {} simulations with various parameters".format(count)
    analysis.to_csv("analysis/analysis.csv")

if __name__ == '__main__':
    # If no arguments supplied, "run" in default mode
    if len(sys.argv) == 1:
        run()
    elif sys.argv[1] == 'analyze':
        analyze(40)
    else:
        print "Invalid arguments!"
