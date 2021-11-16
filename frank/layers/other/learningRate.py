'''
From: https://github.com/TranNhiem/multi_augmentation_strategies_self_supervised_learning/blob/main/losses_optimizers/learning_rate_optimizer_weight_decay_schedule.py

Author: https://github.com/TranNhiem
'''

import tensorflow as tf
import math

'''
********************************************
Training Configure
********************************************
1. Learning Rate
    + particular implementation : Scale Learning Rate Linearly with Batch_SIZE 
    (Warmup: Learning Implementation, and Cosine Anealing + Linear scaling)
   
    # optional not implement yet
    + Schedule Learning with Constrain-Update during training
'''
# Implementation form SimCLR paper (Linear Scale and Sqrt Scale)
# Debug and Visualization
# Section SimCLR Implementation Learning rate BYOL implementation
# https://colab.research.google.com/drive/1MWgcDAqnB0zZlXz3fHIW0HLKwZOi5UBb?usp=sharing

def get_train_steps(num_examples ,args):
    """Determine the number of training steps."""
    if args.train_steps is None:
        train_steps = (num_examples * args.train_epochs //
                       args.train_batch_size + 1)
    else:
        print("You Implement the args training steps")
        train_steps = args.train_steps

    return train_steps

class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule.
    Args:
    Base Learning Rate: is maximum learning Archieve (change with scale applied)
    num_example
    """

    def __init__(self, base_learning_rate, num_examples, args, name=None):
        super(WarmUpAndCosineDecay, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self.args = args
        self._name = name

    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            args = self.args
            warmup_steps = int(
                round(args.warmup_epochs * self.num_examples //
                      args.train_batch_size))
            if args.learning_rate_scaling == 'linear':
                scaled_lr = self.base_learning_rate * args.train_batch_size / 256.
            elif args.learning_rate_scaling == 'sqrt':
                scaled_lr = self.base_learning_rate * \
                    math.sqrt(args.train_batch_size)
            elif args.learning_rate_scaling == 'no_scale':
                scaled_lr = self.base_learning_rate
            else:
                raise ValueError('Unknown learning rate scaling {}'.format(
                    args.learning_rate_scaling))
            learning_rate = (
                step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

            # Cosine decay learning rate schedule
            total_steps = get_train_steps(self.num_examples ,args)
            # TODO(srbs): Cache this object.
            cosine_decay = tf.keras.experimental.CosineDecay(
                scaled_lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate,
                                     cosine_decay(step - warmup_steps))

            return learning_rate