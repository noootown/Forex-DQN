def getConfig():
  return {
    'count': 3,
    'dataType': 'mid',

    # RL
    'episodes': 10000000,
    'interval': 5 * 60 * 1000,

    'learning_rate': 0.9,

    'reward_decay': 0.9,
    'e_greedy': 1.,
    'replace_target_iter': 30000,
    'memory_size': 1500000,
    'batch_size': 32,
    'e_greedy_increment': 0.000001,
    'startStep': 1500,

    'learn_period': 1,

    # network
    'l1_dim': 60,

    'init_w_mean': 0.,
    'init_w_std': 0.01,
    'init_b': 0.01,

    'isTrain': True,
    'isLoad': False, # for continuing previous training

    'optimizer': 'RMSProp',
    'op_rho': None,
    'op_epsilon': None,
    'op_decay': None,
    'op_momentum': 0.95,

    # date
    'dateStart': '20170101',
    'dateEnd': '20170101',

    # output
    'ckptFile': 'train.ckpt',
    'ckptSavePeriod': 10,
    'cliOutput': False,
    'loadHisNum': 0,

    # dir
    'dir': 'tmp',

    'rmHeader': ['rmHeader',
                 'episodes',
                 'isTrain', 'isLoad',
                 'ckptFile', 'ckptSavePeriod',
                 'cliOutput', 'noLoadHeader',
                 'loadHisNum'],

    'noLoadHeader': ['episodes', 'dateStart', 'dateEnd'],
  }
