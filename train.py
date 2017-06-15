from __future__ import print_function

__doc__ = """Training utility functions."""

import numpy as np
import random


def train_epoch(sess, trainable_model, num_iter,
                proportion_supervised, g_steps, d_steps,
                next_sequence, sequences, verify_sequence=None,
                words=None,
                proportion_generated=0.5,
                skipDiscriminator=False):
    """Perform training for model.

    sess: tensorflow session
    trainable_model: the model
    num_iter: number of iterations
    proportion_supervised: what proportion of iterations should the generator
        be trained in a supervised manner (rather than trained via discriminator)
    g_steps: number of generator training steps per iteration
    d_steps: number of discriminator training steps per iteration
    next_sequence: function that returns a groundtruth sequence
    verify_sequence: function that checks a generated sequence, returning True/False
    words:  array of words (to map indices back to words)
    proportion_generated: what proportion of steps for the discriminator
        should be on artificially generated data

    """
    supervised_g_losses = [0]  # we put in 0 to avoid empty slices
    unsupervised_g_losses = [0]  # we put in 0 to avoid empty slices
    d_losses = [0]
    g_loss = None
    d_loss = None
    expected_rewards = [[0] * trainable_model.sequence_length]
    supervised_correct_generation = [0]
    unsupervised_correct_generation = [0]
    supervised_gen_x = None
    supervised_gen_x_attack = None
    unsupervised_gen_x = None
    unsupervised_gen_x_attack = None
    actual_seq = None
    actual_seq_attack = None
    print('running %d iterations with %d g steps and %d d steps' % (num_iter, g_steps, d_steps))
    print('of the g steps, %.2f will be supervised' % proportion_supervised)
    for it in range(num_iter):
        for _ in range(g_steps):
            if random.random() < proportion_supervised:
                seq, seq_attack = next_sequence(sequences)
                actual_seq = seq
                actual_seq_attack = seq_attack
                _, g_loss, g_pred, g_pred_attack = trainable_model.pretrain_step(sess, seq, seq_attack)
                supervised_g_losses.append(g_loss)
                if np.isnan(g_loss):
                    try:
                        print('NAN DEBUG. PRED: ' + str(g_pred))
                    except ValueError:
                        print("VALUE ERROR")
                supervised_gen_x = np.argmax(g_pred, axis=1)
                supervised_gen_x_attack = np.argmax(g_pred_attack, axis=1)
                if verify_sequence is not None:
                    supervised_correct_generation.append(
                        verify_sequence(supervised_gen_x))
            else:
                _, _, g_loss, expected_reward, unsupervised_gen_x, unsupervised_gen_x_attack = \
                    trainable_model.train_g_step(sess)
                expected_rewards.append(expected_reward)
                unsupervised_g_losses.append(g_loss)
                if np.isnan(g_loss):
                    try:
                        print('NAN DEBUG. GEN_X: ' + str(unsupervised_gen_x))
                    except ValueError:
                        print("VALUE ERROR")
                if verify_sequence is not None:
                    unsupervised_correct_generation.append(
                        verify_sequence(unsupervised_gen_x))
        if not skipDiscriminator:
            for _ in range(d_steps):
                if random.random() < proportion_generated:
                    seq,seq_attack = next_sequence(sequences)
                    _, d_loss = trainable_model.train_d_real_step(sess, seq, seq_attack)
                else:
                    _, d_loss = trainable_model.train_d_gen_step(sess)
                d_losses.append(d_loss)

    print('epoch statistics:')
    print('>>>> discriminator loss:', np.mean(d_losses))
    print('>>>> generator loss:', np.mean(supervised_g_losses), np.mean(unsupervised_g_losses))
    if verify_sequence is not None:
        print('>>>> correct generations (supervised, unsupervised):', np.mean(supervised_correct_generation), np.mean(unsupervised_correct_generation))
    print('>>>> actual melody:')
    print(actual_seq,)
    print(actual_seq_attack)
    print('>>>> sampled generations (supervised, unsupervised):',)
    sup_gen_x = [words[x] if words else x for x in supervised_gen_x] if supervised_gen_x is not None else None
    sup_gen_x_attack = [words[x] if words else x for x in supervised_gen_x_attack] if supervised_gen_x_attack is not None else None
    print(sup_gen_x,)
    print(sup_gen_x_attack,)
    unsup_gen_x = [words[x] if words else x for x in unsupervised_gen_x] if unsupervised_gen_x is not None else None
    unsup_gen_x_attack = [words[x] if words else x for x in unsupervised_gen_x_attack] if unsupervised_gen_x_attack is not None else None
    print(unsup_gen_x,)
    print(unsup_gen_x_attack)
    print('>>>> expected rewards:', np.mean(expected_rewards, axis=0))
    return g_loss,d_loss,actual_seq, actual_seq_attack, sup_gen_x, sup_gen_x_attack, unsup_gen_x, unsup_gen_x_attack
