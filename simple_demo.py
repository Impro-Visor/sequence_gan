from __future__ import print_function

__doc__ = """Simple demo on ii-V-I leadsheets.

"""

import model
import train
import json
import numpy as np
import tensorflow as tf
import random

DPATH = "./parsed_ii-V-I_leadsheets/intervalexpert_melodies.json"

NUM_EMB = 37 # Number of possible "characters" in the sequence. Encoding: 0-34 for note vals, 35 for rest, 36 for sustain prev note.
EMB_DIM = 10
HIDDEN_DIM = 70
SEQ_LENGTH = 96
START_TOKEN = 5 # Middle C

EPOCH_ITER = 100
CURRICULUM_RATE = 0.03  # how quickly to move from supervised training to unsupervised
SUP_BASELINE = 0.3 # Decrease ratio of supervised training to this baseline ratio.
TRAIN_ITER = 5000  # generator/discriminator alternating
G_STEPS = 4  # how many times to train the generator each round
D_STEPS = 1  # how many times to train the discriminator per generator steps
LEARNING_RATE = 1e-2 * SEQ_LENGTH
SEED = 88

def get_trainable_model():
    return model.GRU(
        NUM_EMB, EMB_DIM, HIDDEN_DIM,
        SEQ_LENGTH, START_TOKEN,
        learning_rate=LEARNING_RATE)


def verify_sequence(seq):
    """
    Skip verification, assume dataset has valid sequences.
    """
    return True

def get_sequences(datapath):
    """
    Get the training set of sequences.
    """
    sequences = []
    with open(datapath,'r') as datafile:
        sequences = json.load(datafile)
        for i in range(len(sequences)):
            sequences[i] = sequences[i][:SEQ_LENGTH]
    return sequences

def get_random_sequence(sequences):
    """
    Get a random note sequence from training set.
    """
    return random.choice(sequences)


def test_sequence_definition():
    """
    Skip verification, assume dataset has valid sequences.
    """

    #for _ in range(1000):
    #    assert verify_sequence(get_random_sequence())
    pass


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    trainable_model = get_trainable_model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    with open("genx_sup.txt", 'w') as supfile:
        pass

    with open("genx_unsup.txt", 'w') as supfile:
        pass

    with open("genx_act.txt", 'w') as supfile:
        pass

    print('training')
    actuals = []
    sups = []
    unsups = []
    for epoch in range(TRAIN_ITER // EPOCH_ITER):
        print('epoch', epoch)
        proportion_supervised = max(SUP_BASELINE, 1.0 - CURRICULUM_RATE * epoch)
        actual_seq, sup_gen_x, unsup_gen_x = train.train_epoch(
            sess, trainable_model, EPOCH_ITER,
            proportion_supervised=proportion_supervised,
            g_steps=G_STEPS, d_steps=D_STEPS,
            next_sequence=get_random_sequence,
            sequences=get_sequences(DPATH),
            verify_sequence=None)
        actuals.append(actual_seq)
        sups.append(sup_gen_x)
        unsups.append(unsup_gen_x)
    all_seqs = [actuals, sups, unsups]
    with open("generations.json",'w') as dumpfile:
        json.dump(all_seqs, dumpfile)

if __name__ == '__main__':
    test_sequence_definition()
    main()
