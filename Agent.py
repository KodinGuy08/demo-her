import os, time
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.compat.v1.initializers import random_uniform

import DataGen as DG

tf.disable_v2_behavior()
tf.disable_eager_execution()

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, n_goals):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.goal_memory = np.zeros((self.mem_size, *n_goals))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done, goal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.goal_memory[index] = goal
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        goal = self.goal_memory[batch]

        return states, actions, rewards, states_, terminal, goal

class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, goal_dims, sess, fc_dims,
                 action_bound, batch_size=64, chkpt_dir='tmp2/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name

        self.fc_dims = fc_dims

        self.chkpt_dir = chkpt_dir

        self.input_dims = input_dims
        self.goal_dims = goal_dims

        self.batch_size = batch_size
        self.sess = sess
        self.action_bound = action_bound
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(
            self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        self.optimize = tf.train.AdamOptimizer(self.lr).\
                    apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')
            self.goal_in = tf.placeholder(tf.float32,
                                        shape=[None, *self.goal_dims],
                                        name='goals')

            self.action_gradient = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='gradients')

            concat_input = tf.concat([self.input, self.goal_in], axis=1)

            f1 = 1. / np.sqrt(self.fc_dims[0])

            dense1 = tf.layers.dense(concat_input, units=self.fc_dims[0],
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1. / np.sqrt(self.fc_dims[1])

            dense2 = tf.layers.dense(layer1_activation, units=self.fc_dims[1],
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.relu(batch2)

            f3 = 0.003

            mu = tf.layers.dense(layer2_activation, units=self.n_actions,
                            activation='tanh',
                            kernel_initializer= random_uniform(-f3, f3),
                            bias_initializer=random_uniform(-f3, f3))
            self.mu = tf.multiply(mu, self.action_bound)

    def predict(self, inputs, goal):
        return self.sess.run(self.mu, feed_dict={
            self.input: inputs,
            self.goal_in: goal
        })

    def train(self, inputs, goal, gradients):
        self.sess.run(self.optimize,
                      feed_dict={
                          self.input: inputs,
                          self.goal_in: goal,
                          self.action_gradient: gradients
                      })

    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, goal_dims, sess, fc_dims,
                 batch_size=64, chkpt_dir='tmp2/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name

        self.fc_dims = fc_dims

        self.chkpt_dir = chkpt_dir

        self.input_dims = input_dims
        self.goal_dims = goal_dims

        self.batch_size = batch_size
        self.sess = sess
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name +'_ddpg.ckpt')

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')
            self.goal_in = tf.placeholder(tf.float32,
                                        shape=[None, *self.goal_dims],
                                        name='goals')

            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='actions')

            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None,1],
                                           name='targets')

            concat_input = tf.concat([self.input, self.goal_in], axis=1)

            f1 = 1. / np.sqrt(self.fc_dims[0])
            dense1 = tf.layers.dense(concat_input, units=self.fc_dims[0],
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1. / np.sqrt(self.fc_dims[1])
            dense2 = tf.layers.dense(layer1_activation, units=self.fc_dims[1],
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.relu(batch2)

            #layer2_activation = tf.nn.relu(batch2)
            #layer2_activation = tf.nn.relu(dense2)

            action_in = tf.layers.dense(self.actions, units=self.fc_dims[1],
                                        activation='relu')
            #batch2 = tf.nn.relu(batch2)
            # no activation on action_in and relu activation on state_actions seems to
            # perform poorly.
            # relu activation on action_in and relu activation on state_actions
            # does reasonably well.
            # relu on batch2 and relu on action in performs poorly

            #state_actions = tf.concat([layer2_activation, action_in], axis=1)
            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.relu(state_actions)
            f3 = 0.003
            self.q = tf.layers.dense(state_actions, units=1,
                               kernel_initializer=random_uniform(-f3, f3),
                               bias_initializer=random_uniform(-f3, f3),
                               kernel_regularizer=tf.keras.regularizers.l2(0.01))

            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, goal, actions):
        return self.sess.run(self.q,
                             feed_dict={self.input: inputs,
                                        self.goal_in: goal,
                                        self.actions: actions})
    def train(self, inputs, goal, actions, q_target):
        return self.sess.run(self.optimize,
                      feed_dict={self.input: inputs,
                                 self.goal_in: goal,
                                 self.actions: actions,
                                 self.q_target: q_target})

    def get_action_gradients(self, inputs, goal, actions):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.input: inputs,
                                        self.goal_in: goal,
                                        self.actions: actions})
    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

class Agent(object):
    def __init__(self, alpha, beta, input_dims, goal_dims, tau, env, gamma=0.99, n_actions=2,
                 max_size=1000000, layer_size=[400, 300],
                 batch_size=64, chkpt_dir='tmp/ddpg', Datagen=False, load=True):
        self.env = env
        self.epsilon = 0

        self.gamma = gamma
        self.tau = tau

        self.memory = ReplayBuffer(max_size, input_dims, n_actions, goal_dims)
        self.demo_mem = ReplayBuffer(max_size, input_dims, n_actions, goal_dims)
        if Datagen:
            self.datagen()

        self.batch_size = batch_size
        self.sess = tf.Session()
        self.actor = Actor(alpha, n_actions, 'Actor', input_dims, goal_dims, self.sess,
                           layer_size, env.action_space.high,
                            chkpt_dir=chkpt_dir)
        self.critic = Critic(beta, n_actions, 'Critic', input_dims, goal_dims, self.sess,
                             layer_size, chkpt_dir=chkpt_dir)

        self.target_actor = Actor(alpha, n_actions, 'TargetActor',
                                  input_dims, goal_dims,
                                  self.sess, layer_size, env.action_space.high,
                                  chkpt_dir=chkpt_dir)
        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims,
                                    goal_dims,
                                    self.sess, layer_size,
                                    chkpt_dir=chkpt_dir)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # define ops here in __init__ otherwise time to execute the op
        # increases with each execution.
        self.update_critic = \
        [self.target_critic.params[i].assign(
                      tf.multiply(self.critic.params[i], self.tau) \
                    + tf.multiply(self.target_critic.params[i], 1. - self.tau))
         for i in range(len(self.target_critic.params))]

        self.update_actor = \
        [self.target_actor.params[i].assign(
                      tf.multiply(self.actor.params[i], self.tau) \
                    + tf.multiply(self.target_actor.params[i], 1. - self.tau))
         for i in range(len(self.target_actor.params))]

        self.sess.run(tf.global_variables_initializer())

        self.update_network_parameters(first=True)

        if load:
            self.load_models()

        #if Datagen:
        #    for i in range(124):
        #        self.learn()

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done, goal):
        self.memory.store_transition(state, action, reward, new_state, done, goal)

    def choose_action(self, state, goal):
        state = state[np.newaxis, :]
        goal = goal[np.newaxis, :]
        mu = self.actor.predict(state, goal) # returns list of list
        noise = self.noise()
        mu_prime = mu + self.epsilon #* noise

        return mu_prime[0]

    def learn(self):
        demo_batch_size = 196
        state, action, reward, new_state, done, goal = [None]*6

        timer = 0

        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done, goal = \
            self.memory.sample_buffer(self.batch_size - demo_batch_size)

        dem_size = demo_batch_size
        stated, actiond, rewardd, new_stated, doned, goald = self.demo_mem.sample_buffer(dem_size)

        def combine(in1, in2):
            return np.concatenate([in1.copy(), in2.copy()])

        state = combine(state, stated)
        action = combine(action, actiond)
        reward = combine(reward, rewardd)
        new_state = combine(new_state, new_stated)
        done = combine(done, doned)
        goal = combine(goal, goald)

        timer = time.time()
        critic_value_ = self.target_critic.predict(new_state, goal,
                                           self.target_actor.predict(new_state, goal))
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = np.reshape(target, (self.batch_size, 1))


        print("\n Post 1:", timer - time.time())
        timer = time.time()
        
        _ = self.critic.train(state, goal, action, target)
        
        print("\n Post 2:", timer - time.time())
        timer = time.time()

        a_outs = self.actor.predict(state, goal)
        grads = self.critic.get_action_gradients(state, goal, a_outs)

        self.actor.train(state, goal, grads[0])
        
        print("\n Post 3:", timer - time.time())

        self.update_network_parameters()

        #self.epsilon = self.epsilon - self.epsilon_D
        #self.noise.reset()

    def eval(self):
        q_val = []
        for i in range(1):
            res = self.env.reset()
            new_state = res['observation']
            goal = res['desired_goal']
            agoal = res['achieved_goal']

            for t in range(28):
                ng = np.array(agoal) - np.array(goal)

                a = self.actor.predict(new_state, ng)
                new_state, reward, done, info = self.env.step(a)

                critic_value_ = self.critic.predict(new_state, ng,
                                    a)
                q_val.append(critic_value_)
        return q_val

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def subtract_array(self, a, b):
        c = []
        for i in range(0, len(a)):
            c.append(a[i] - b[i])
        return b

    def datagen(self):
        actions, observations, infos, rewards = DG.main()
        for ep in range(0, len(actions)):
            acts, obs, ins, rs = actions[ep], observations[ep], infos[ep], rewards[ep]
            for step in range(0, len(acts) - 1):
                ob = obs[step]['observation'].copy()
                act = acts[step]
                reward = rs[step]
                ob_ = obs[step+1]['observation'].copy()

                goal = np.array(obs[step]['desired_goal']).copy()
                g = np.array(obs[step]['achieved_goal']).copy()

                #print("\n\n", "## LOG ##", "\n", goal, "\n", g, "\n", self.subtract_array(g, goal))
                self.demo_mem.store_transition(ob, act, reward, ob_, 1, self.subtract_array(g, goal))

