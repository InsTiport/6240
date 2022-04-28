from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from doctest import testfile
from inspect import istraceback

import tensorflow as tf
import pandas as pd
from os import path
import numpy as np
import model
import sys


flags = tf.app.flags
flags.DEFINE_string('train', 'data', 'Path to put the training data.')
flags.DEFINE_string('test', 'data', 'Path to put the testing data.')
flags.DEFINE_string('ckpt', 'ckpt', 'Directory to put the ckpg data.')
flags.DEFINE_string('metrics', 'recall', 'Metrics Type')


FLAGS = flags.FLAGS


class HGRU4RecTest(tf.test.TestCase):

  def setUp(self):
    print('setup is called')


  def testBuildModel(self):
    train_data = pd.read_hdf(FLAGS.train, 'train')
    test_data = pd.read_hdf(FLAGS.test, 'test')
    itemids = train_data['item_id'].unique()
    item_test = test_data['item_id'].unique()


    n_items = 855
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
      hgru4rec = model.HGRU4Rec(sess, [172], [100], batch_size=256, n_items=n_items,
                                checkpoint_dir=FLAGS.ckpt,
                                log_dir=r'./log',
                                session_key='session_id',
                                item_key='item_id',
                                time_key='created_at',
                                user_key='user_id',
                                is_training=False,
                                dropout_p_hidden_usr=1.0,
                                dropout_p_hidden_ses=1.0,
                                dropout_p_init = 0,
                                n_epochs=1,
                                test_metrics=FLAGS.metrics
                                )

      hgru4rec.fit(train_data, test_data, test_data, is_training=False, use_embedding=True)


if __name__ == "__main__":
  tf.test.main()
