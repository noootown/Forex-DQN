import numpy as np
import tensorflow as tf
import datetime as dt

from account import Account
from constants import STOP, BUY, SELL, CLOSE, SHOW_HAND
from helper import mkdir

np.random.seed(dt.datetime.now().microsecond)
tf.set_random_seed(dt.datetime.now().microsecond)

class DeepNetwork:
  def __init__(
    self,
    forex,
    dates,
    featureNum,
    config,
  ):
    self.forex = forex
    self.dates = dates
    self.config = config

    self.episodes = config['episodes']
    self.interval = config['interval']

    self.n_actions = 4
    self.n_features = featureNum * config['count']
    self.lr = config['learning_rate']
    self.gamma = config['reward_decay']
    self.epsilon_max = config['e_greedy'] if config['isTrain'] else 1
    self.replace_target_iter = config['replace_target_iter']
    self.memory_size = config['memory_size']
    self.batch_size = config['batch_size']
    self.epsilon_increment = config['e_greedy_increment']
    self.epsilon = 0 if config['e_greedy_increment'] is not None else self.epsilon_max

    # account
    self.initBalance = 100000

    self.isTrain = config['isTrain']
    self.dir = config['dir']

    self.ckptFile = config['ckptFile']
    self.ckptSavePeriod = config['ckptSavePeriod']

    # total learning step
    self.step = -config['startStep']
    self.totalLoss = 0
    self.totalMaxQ = 0
    self.r_actions = []

    # initialize zero memory [s, a, r, s_]
    self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
    self.memory_counter = 0

    self.sess = tf.Session()

    # consist of [target_net, evaluate_net]
    self.buildNet()

    self.saver = tf.train.Saver(max_to_keep = int(self.episodes / self.ckptSavePeriod))

    if config['isLoad'] or not self.isTrain:
      self.saver.restore(self.sess, 'data/%s/%s' % (self.dir, self.ckptFile))
      print('Load data/%s/%s sucessfully!\n' % (self.dir, self.ckptFile))
    else:
      self.sess.run(tf.global_variables_initializer())
      print('Apply global initializer!\n')

  def subTrain(self, isTrain, dates):
    account = Account(
      balance = self.initBalance,
      cliOutput = self.config['cliOutput'],
    )
    for date in dates:
      self.forex.setDate(date)

      startTime, endTime = self.forex.getTime()
      price, state = self.forex.getPrice(startTime)

      for time in range(startTime, endTime - self.interval, self.interval):
        # Q learning start
        action = self.chooseAction(state, isTrain)

        if not isTrain:
          print(action)

        reward = 0
        if action == STOP:
          reward = account.stop()
        elif action == BUY or action == SELL:
          reward = account.order(
            price, # price is at column 0
            {
              'type': action,
              'unit': SHOW_HAND,
            },
            time = time
          )
        elif action == CLOSE:
          reward = account.closePosition(price, time = time)

        price, state_ = self.forex.getPrice(time + self.interval)

        self.storeTransition(
          transition = np.hstack((state, [action, reward], state_)),
          mode = 0 if isTrain else 1,
        )

        if isTrain:
          if self.step > 0 and self.step % self.config['learn_period'] == 0:
            self.learn()
          self.step += 1

        state = state_

    return account.balance

  def train(self):
    if self.isTrain:
      print('Start training\n')
    else:
      print('Start testing\n')

    for episode in range(self.episodes):
      if episode % 10 == 0:
        print('episode', episode)

      if self.isTrain:
        epsilonBalance = self.subTrain(isTrain = True, dates = self.dates)

        # to get the actual balance
        realBalance = self.subTrain(isTrain = False, dates = self.dates)

        self.finishEpisode(episode, epsilonBalance, realBalance)
      else:
        if self.config['cliOutput']:
          print(Account.getCloseHeader())

        print(self.subTrain(isTrain = False, dates = self.dates))

    if self.isTrain:
      print('Finish training\n')
    else:
      print('Finish testing\n')

  def optimize(self):
    if self.config['optimizer'] == 'RMSProp':
      return tf.train.RMSPropOptimizer(
        self.lr,
        decay = 0.9 if self.config['op_decay'] == None else self.config['op_decay'],
        momentum = 0.0 if self.config['op_momentum'] == None else self.config['op_momentum'],
        epsilon = pow(10, -10) if self.config['op_epsilon'] == None else self.config['op_epsilon'],
      ).minimize(self.loss)

  def buildNet(self):
    def addLayer(
            name,
            input,
            output_dim,
            w_init,
            b_init,
            c_names,
            active_fn = None,
    ):
      with tf.variable_scope(name):
        w = tf.get_variable('w', [input.get_shape().as_list()[1], output_dim],
                            initializer = w_init, collections = c_names)
        b = tf.get_variable('b', [1, output_dim], initializer = b_init, collections = c_names)

        if active_fn != None:
          out = active_fn(tf.matmul(input, w) + b)
        else:
          out = tf.matmul(input, w) + b

      return out, w, b

    # ------------------ build evaluate_net ------------------
    self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
    self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

    # config of layers
    self.w = {}

    self.w_init = \
      tf.random_normal_initializer(self.config['init_w_mean'], self.config['init_w_std'])
    self.b_init = \
      tf.constant_initializer(self.config['init_b'])

    active_fn = tf.nn.relu

    with tf.variable_scope('eval_net'):
      # c_names(collections_names) are the collections to store variables
      c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

      self.w['l1_o'], self.w['l1_w'], self.w['l1_b'] =\
        addLayer('l1', self.s, self.config['l1_dim'], self.w_init, self.b_init, c_names, active_fn)

      # output layer
      self.q_eval, self.w['lout_w'], self.w['lout_b'] = \
        addLayer('lout', self.w['l1_o'], self.n_actions, self.w_init, self.b_init, c_names)

    with tf.name_scope('loss'):
      self.loss = tf.reduce_sum(tf.squared_difference(self.q_target, self.q_eval))

    with tf.name_scope('train'):
      self._train_op = self.optimize()

    # ------------------ build target_net ------------------
    self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
    self.t_w = {}

    with tf.variable_scope('target_net'):
      c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

      self.t_w['l1_o'], self.t_w['l1_w'], self.t_w['l1_b'] = \
        addLayer('l1', self.s_, self.config['l1_dim'], self.w_init, self.b_init, c_names, active_fn)

      # output layer
      self.q_next, self.t_w['lout_w'], self.t_w['lout_b'] = \
        addLayer('lout', self.t_w['l1_o'], self.n_actions, self.w_init, self.b_init, c_names)

    with tf.variable_scope('summary'):
      # e_XX means with epsilon
      # r_XX means without epsilon, which is real simulation
      scalar_summary_tags = ['loss_avg', 'e_balance', 'r_balance',
                             'q_max', 'q_total', 'epsilon']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_') + '_0')
        self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])

      histogram_summary_tags = ['r_actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_') + '_0')
        self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])

    with tf.variable_scope('param'):
      w_c_names = 'eval_net_params_summaries'
      histogram_w_tags = ['l1_w', 'l1_b', 'lout_w', 'lout_b']
      for tag in histogram_w_tags:
        tf.summary.histogram(tag, self.w[tag], collections = [w_c_names])

    if self.isTrain:
      self.merged = tf.summary.merge_all(key = w_c_names)
      self.writer = tf.summary.FileWriter('data/' + self.dir, self.sess.graph)

  # state action reward next state
  # mode: 0: store train 1: store test
  def storeTransition(self, transition, mode = 0):
    # replace the old memory with new memory
    self.memory[self.memory_counter % self.memory_size] = transition
    self.memory_counter += 1

  def chooseAction(self, observation, isTrain = True):
    # to have batch dimension when feed into tf placeholder
    observation = observation[np.newaxis, :]

    if not isTrain:
      actions = self.sess.run(self.q_eval, feed_dict={self.s: observation})
      action = np.argmax(actions)
      self.r_actions.append(action)

    elif self.step < 0 or np.random.uniform() >= self.epsilon:
      action = np.random.randint(0, self.n_actions)

    else:
      # forward feed the observation and get q value for every actions
      actions = self.sess.run(self.q_eval, feed_dict={self.s: observation})
      action = np.argmax(actions)

    return action

  def replaceTargetParams(self):
    t_params = tf.get_collection('target_net_params')
    e_params = tf.get_collection('eval_net_params')
    self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

  def learn(self):
    # check to replace target parameters
    if self.step % self.replace_target_iter == 0:
      self.replaceTargetParams()

    # sample batch memory from all memory
    batch_memory =\
    self.memory[np.random.choice(
      self.memory_size\
      if self.memory_counter > self.memory_size\
      else self.memory_counter,
      self.batch_size), :]

    q_next, q_eval = self.sess.run(
      [self.q_next, self.q_eval],
      feed_dict={
        self.s_: batch_memory[:, -self.n_features:],
        self.s: batch_memory[:, :self.n_features]
      })

    # change q_target w.r.t q_eval's action
    q_target = q_eval.copy()

    q_target[np.arange(self.batch_size), batch_memory[:, self.n_features].astype(int)] = \
      batch_memory[:, self.n_features + 1] + self.gamma * np.max(q_next, axis=1)

    # train eval network
    _, self.param_summary, cost = \
      self.sess.run([self._train_op, self.merged, self.loss],
                    feed_dict={self.s: batch_memory[:, :self.n_features],
                               self.q_target: q_target,
                               })

    # increasing epsilon
    self.epsilon =\
      self.epsilon + self.epsilon_increment\
      if self.epsilon < self.epsilon_max\
      else self.epsilon_max
    self.totalLoss += cost
    self.totalQ += q_eval.mean(axis = 1).mean(axis = 0)
    self.totalMaxQ += np.max(q_eval, axis=1).mean()

  # mode 0: normal save, 1: period save
  def saveParam(self, dir = 'tmp', mode = 0):
    subdir = ''

    if mode == 1:
      subdir = 'history/%s/' % (dir)

    fulldir = 'data/%s/%s' % (self.dir, subdir)

    mkdir(fulldir)
    self.saver.save(self.sess, '%s%s' % (fulldir, self.ckptFile))

  def injectSummary(self, tag_dict, episode):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, episode)

    self.writer.add_summary(self.param_summary, episode)

  def finishEpisode(self, episode, epsilonBalance, realBalance):
    if self.step > 0:
      injectDict = {
        # scalar
        'loss_avg': self.totalLoss,
        'e_balance': epsilonBalance,
        'r_balance': realBalance,
        'epsilon': self.epsilon,
        'q_max': self.totalMaxQ,
        'q_total': self.totalQ,
        # histogram
        'r_actions': self.r_actions,
      }
      self.injectSummary(injectDict, episode)

      self.saveParam(mode = 0)
      if episode % self.ckptSavePeriod == 0:
        self.saveParam(dir = '%d' % (episode), mode = 1)

    self.r_actions = []
    self.totalLoss = 0
    self.totalQ = 0
    self.totalMaxQ = 0
