import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class QLearner(object):
    """TODO: Fill this later."""

    discount_factor = 0.8

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.qtable = dict()

        # Q-Learner needs to span how many states and actions?
        total_num_states = 1
        for attribute in self.state_space.iterkeys():
            total_num_states *= len(self.state_space[attribute])
        print "Q-Learner has {} possible states and {} actions".format(total_num_states, len(self.action_space))

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

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # Initialize additional variables here
        self.qlearner = QLearner(self.state_space, self.allowed_actions)
        self.prev_state = None
        self.state = None
        self.cumulative_reward = 0
        self.exploration_factor = 1.0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required
        self.prev_state = None
        self.state = None
        self.cumulative_reward = 0
        self.exploration_factor *= 0.9

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = {
            'next_waypoint': self.next_waypoint, 
            'oncoming': inputs['oncoming'],
            'left': inputs['left'],
            'light': inputs['light']
        }
        
        # Select action according to your policy
        if random.random() > self.exploration_factor:
            action = self.qlearner.optimal_action(self.state)
        else:
            action = random.choice(self.allowed_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        if self.prev_state is not None:
            self.qlearner.update(self.state, action, reward, self.prev_state, 1./t)
        
        self.prev_state = self.state

        self.cumulative_reward += reward
        #print "Cumulative reward is {}".format(self.cumulative_reward)
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.05, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
