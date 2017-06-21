from __future__ import print_function

__doc__ = """Simple demo on ii-V-I leadsheets.

"""

import model
import train
import json
import numpy as np
import tensorflow as tf
import random

DPATH = "./parsed_ii-V-I_leadsheets/pitchexpert_melodies.json"
CPATH = "./parsed_ii-V-I_leadsheets/pitchexpert_chords.json"
PPATH = "./parsed_ii-V-I_leadsheets/pitchexpert_pos.json"

NUM_EMB = 36 # Number of possible "characters" in the sequence. Encoding: 0-34 for note vals, 35 for rest.
NUM_EMB_ATTACK = 2
EMB_DIM = 10
EMB_DIM_ATTACK = 1
HIDDEN_DIM = 300
HIDDEN_DIM_ATTACK = 50
SEQ_LENGTH = 96
START_TOKEN = 35 # Middle C
START_TOKEN_ATTACK = 0
START_TOKEN_POS_LOW = 0
START_TOKEN_POS_HIGH = 0

EPOCH_ITER = 100
CURRICULUM_RATE = 0.1  # how quickly to move from supervised training to unsupervised
SUP_BASELINE = 0.0 # Decrease ratio of supervised training to this baseline ratio.
TRAIN_ITER = 3000  # generator/discriminator alternating
G_STEPS = 7  # how many times to train the generator each round
D_STEPS = 1  # how many times to train the discriminator per generator steps
LEARNING_RATE = 1e-3 * SEQ_LENGTH
SEED = 88

def get_trainable_model():
    return model.GRU(
        NUM_EMB, NUM_EMB_ATTACK, EMB_DIM, EMB_DIM_ATTACK, HIDDEN_DIM, HIDDEN_DIM_ATTACK,
        SEQ_LENGTH, START_TOKEN, START_TOKEN_ATTACK, START_TOKEN_POS_LOW, START_TOKEN_POS_HIGH,
        learning_rate=LEARNING_RATE)


def verify_sequence(seq):
    """
    Skip verification, assume dataset has valid sequences.
    """
    return True

def get_sequences(datapath,chordpath,pospath):
    """
    Get the training set of sequences.
    """
    sequences = []
    with open(datapath,'r') as datafile:
        sequences = json.load(datafile)
        for i in range(len(sequences)):
            sequences[i] = sequences[i][:SEQ_LENGTH]
    chordseqs = []
    with open(chordpath,'r') as chordfile:
        chordseqs = json.load(chordfile)
        for i in range(len(chordseqs)):
            chordseqs[i] = chordseqs[i][:SEQ_LENGTH]
    lows = []
    highs = []
    with open(pospath, 'r') as posfile:
        posseqs = json.load(posfile)
        for posseq in posseqs:
            low = [x[0] for x in posseq[:SEQ_LENGTH]]
            high = [x[1] for x in posseq[:SEQ_LENGTH]]
            lows.append(low)
            highs.append(high)

    return sequences,chordseqs,lows,highs

def get_random_sequence(sequences,chordseqs,lows,highs):
    """
    Get a random note sequence from training set.
    """
    i = random.randint(0,len(sequences)-1)
    sequence = sequences[i]
    chordseq = chordseqs[i]
    chordkeys = [x[0] for x in chordseq]
    chordnotes = [x[1] for x in chordseq]
    low = lows[i]
    high = highs[i]
    notes = [x[0] for x in sequence]
    attacks = [x[1] for x in sequence]
    return notes,attacks,chordkeys,chordnotes,low,high


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
    curric_count = 0
    proportion_supervised = 1.0
    skipD = False
    skipG = False
    startUnsup = False
    for epoch in range(TRAIN_ITER // EPOCH_ITER):
        print('epoch', epoch)
        melodyseqs,chordseqs,lows,highs = get_sequences(DPATH,CPATH,PPATH)
        latest_g_loss,latest_d_loss,actual_seq, actual_seq_attack, sup_gen_x, sup_gen_x_attack, unsup_gen_x, unsup_gen_x_attack = train.train_epoch(
            sess, trainable_model, EPOCH_ITER,
            proportion_supervised=proportion_supervised,
            g_steps=G_STEPS, d_steps=D_STEPS,
            next_sequence=get_random_sequence,
            sequences=melodyseqs,chordseqs=chordseqs,lows=lows,highs=highs,
            verify_sequence=None,skipDiscriminator = skipD,skipGenerator = skipG)
        actuals.append([actual_seq,actual_seq_attack])
        sups.append([sup_gen_x,sup_gen_x_attack])
        unsups.append([unsup_gen_x,unsup_gen_x_attack])
        if not startUnsup and latest_d_loss != None and latest_d_loss < 0.9:
            print('###### FREEZING DISCRIMINATOR')
            skipD = True
        if latest_g_loss != None and latest_g_loss < 2:
            startUnsup = True
        if startUnsup:
            skipG = False
            skipD = False
            if latest_d_loss != None and latest_d_loss > 90.7:
                print('###### FREEZING GENERATOR')
                skipG = True
                continue
            curric_count+=1
            proportion_supervised = max(SUP_BASELINE, 0.3 - CURRICULUM_RATE * curric_count)

    all_seqs = [actuals, sups, unsups]
    for seqs in all_seqs:
        for seq in seqs:
            if seq != None:
                for val in seq:
                    if val != None:
                        for i in range(len(val)):
                            if val[i] != None:
                                val[i] = int(val[i])
    with open("generations.json",'w') as dumpfile:
        json.dump(all_seqs, dumpfile)

if __name__ == '__main__':
    test_sequence_definition()
    main()
