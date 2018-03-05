#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:03:49 2018

@author: celestinoedano
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import shift_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default = 100, type = int, help = 'batch size')
parser.add_argument('--train_steps', default = 1000, type = int,
                    help = 'number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])
    (train_x, train_y), (test_x, test_y) = shift_data.load_data()
    
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.nukmeric_column(key = key))
        
    classifier = tf.estimator.DNNClassifier(
            feature_columns = my_feature_columns,
            hidde_units = [10, 10],
            n_classes = 11)
    
    classifier.train(
            input_fn = lambda:shift_data.train_input_fn(train_x, train_y,
                                                        args.batch_size),
            steps = args.train_steps)
            
    eval_result = classifier.evaluate(
            input_fn = lambda:shift_data.eval_input_fn(test_x, test_y,
                                                       args.batch_size))
                                                       
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    
    expected = ['OT15', 'OT20', 'OP20']
    predict_x = {
            'type': [1, 1, 0],
            'clock_in': [0.25, 0.36, 0.58],
            'clock_in_day': [4, 4, 0],
            'break': [1, 0, 0],
            'clock_out': [0.63, 0.75, 0.92],
            'clock_out_day': [4, 4, 0],
            'total_working_hrs': [0.36, 0.38, 0.33],
            'holiday': [0, 0, 0],
            'award': [5, 6, 3],
            }
    
    predictions = classifier.predict(
            input_fn = lambda:shift_data.eval_input_fn(predict_x,
                                                        labels = None,
                                                        batch_size = args.atch_size))
    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        
        print(template.format(shift_data.award[class_id],
                              100 * probability, expec))
        
    if __name__ == '__main__':
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.app.run(main)