from __future__ import print_function

__doc__ = """Simple demo on ii-V-I leadsheets.

"""

import model
import train
import json
import numpy as np
import tensorflow as tf
import random

TWOFIVEONE = 1
TRANSCRIPTIONS = 2
LEADSHEET_CHOICE = TWOFIVEONE

PITCH = 0
INTERVAL = 1
CHORD = 2
EXPERT = PITCH

ONEHOT = 0
BIT = 1
ENCODING = ONEHOT

USING_DURATIONS = True

parsename = ""
if LEADSHEET_CHOICE == TWOFIVEONE:
    print("Using ii-V-I Leadsheets")
    parsename = "ii-V-I_leadsheets"
elif LEADSHEET_CHOICE == TRANSCRIPTIONS:
    print("Using Transcriptions")
    parsename = "transcriptions"

expertname = ""
NOTEADJUST = 0
if EXPERT == PITCH:
    print("Expert: PITCH")
    expertname = "pitch"
elif EXPERT == INTERVAL:
    print("Expert: INTERVAL")
    expertname = "interval"
    NOTEADJUST = 13
elif EXPERT == CHORD:
    print("Expert: CHORD")
    expertname = "chord"

encodingname = ""
if ENCODING == ONEHOT:
    print("Encoding: ONEHOT")
    encodingname = "onehot"
elif ENCODING == BIT:
    print("Encoding: BIT")
    encodingname = "bit"

dur_append = ""
if USING_DURATIONS:
    print("Durations: ON")
    dur_append = "_dur"

NPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_melodies.json"
DPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_durs.json"
CPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_chords.json"
PPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_pos.json"
SPPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_startpitches.json"
print(NPATH)
print(DPATH)
print(CPATH)
print(PPATH)
print(SPPATH)

NUM_EMB = -1
MIDI_MIN = -1
MIDI_MAX = -1
START_TOKEN = -1
if EXPERT == PITCH:
    NUM_EMB = 36 # Number of possible "characters" in the sequence. Encoding: 0-34 for note vals, 35 for rest.
    MIDI_MIN = 55 # lowest note value found in trainingset
    MIDI_MAX = 89 # highest note value found in trainingset
    START_TOKEN = 5 # middle C
elif EXPERT == INTERVAL:
    NUM_EMB = 29 # Min int = -13, Max int = 14. 0-27 for jumps, 28 for rest.
    MIDI_MIN = -13 # lowest interval value found in trainingset
    MIDI_MAX = 14 # highest interval value found in trainingset
    START_TOKEN = 0
if LEADSHEET_CHOICE == TRANSCRIPTIONS:
    NUM_EMB = 64
    MIDI_MIN = 44
    MIDI_MAX = 106
    START_TOKEN = 0
NUM_EMB_DUR = 15
EMB_DIM = 10
EMB_DIM_DUR = 5
HIDDEN_DIM = 300
HIDDEN_DIM_DUR = 50
MAX_SEQ_LENGTH = 96
if LEADSHEET_CHOICE == TWOFIVEONE:
    MAX_SEQ_LENGTH = 27
elif LEADSHEET_CHOICE == TRANSCRIPTIONS:
    MAX_SEQ_LENGTH = 1000
START_TOKEN_DUR = 0
START_TOKEN_POS_LOW = 0
START_TOKEN_POS_HIGH = 0

EPOCH_ITER = 100
CURRICULUM_RATE = 0.1  # how quickly to move from supervised training to unsupervised
SUP_BASELINE = 0.0 # Decrease ratio of supervised training to this baseline ratio.
TRAIN_ITER = 15000  # generator/discriminator alternating
G_STEPS = 7  # how many times to train the generator each round
D_STEPS = 1  # how many times to train the discriminator per generator steps
LEARNING_RATE = 1e-3 * MAX_SEQ_LENGTH
SEED = 88

def get_trainable_model():
    return model.GRU(
        NUM_EMB, NUM_EMB_DUR, EMB_DIM, EMB_DIM_DUR, HIDDEN_DIM, HIDDEN_DIM_DUR,
        MAX_SEQ_LENGTH, START_TOKEN, START_TOKEN_DUR, START_TOKEN_POS_LOW, START_TOKEN_POS_HIGH,
        learning_rate=LEARNING_RATE,MIDI_MIN=MIDI_MIN,MIDI_MAX=MIDI_MAX)


def verify_sequence(seq):
    """
    Skip verification, assume dataset has valid sequences.
    """
    return True

def get_sequences(notepath,durpath,chordpath,pospath,startppath):
    """
    Get the training set of sequences.
    """
    noteseqs = []
    with open(notepath,'r') as notefile:
        noteseqs = json.load(notefile)
        for i in range(len(noteseqs)):
            noteseqs[i] = noteseqs[i]#[:SEQ_LENGTH]
    durseqs = []
    with open(durpath,'r') as durfile:
        durseqs = json.load(durfile)
        #for i in range(len(durseqs)):
        #    durseqs[i] = durseqs[i]#[:SEQ_LENGTH]
    chordseqs = []
    with open(chordpath,'r') as chordfile:
        chordseqs = json.load(chordfile)
        #for i in range(len(chordseqs)):
        #    chordseqs[i] = chordseqs[i]#[:SEQ_LENGTH]
    lows = []
    highs = []
    with open(pospath, 'r') as posfile:
        posseqs = json.load(posfile)
        for posseq in posseqs:
            low = [x[0] for x in posseq]#posseq[:SEQ_LENGTH]]
            high = [x[1] for x in posseq]#posseq[:SEQ_LENGTH]]
            lows.append(low)
            highs.append(high)
    spseq = []
    with open(startppath, 'r') as spfile:
        spseq = json.load(spfile)

    return noteseqs,durseqs,chordseqs,lows,highs,spseq

def get_random_sequence(sequences,durseqs,chordseqs,lows,highs,spseq):
    """
    Get a random note sequence from training set.
    """
    i = random.randint(0,len(sequences)-1)
    notes = sequences[i]
    durs = durseqs[i]
    chordseq = chordseqs[i]
    chordkeys = np.array([x[0] for x in chordseq])
    #chordkeys_temp = np.zeros((len(chordkeys),12)) # onehot encode the chord key
    #chordkeys_temp[np.arange(len(chordkeys)),chordkeys] = 1
    #chordkeys_temp = list(chordkeys_temp)
    chordkeys_onehot = [x[0] for x in chordseq]#[list(x) for x in chordkeys_temp]
    chordkeys = [x[0] for x in chordseq]
    chordnotes = [x[1] for x in chordseq]
    low = lows[i]
    high = highs[i]
    sequence_length = len(notes)
    start_pitch = spseq[i]
    return notes,durs,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,start_pitch


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
    generations = []
    curric_count = 0
    proportion_supervised = 1.0
    skipD = False
    skipG = False
    startUnsup = False
    for epoch in range(TRAIN_ITER // EPOCH_ITER):
        print('epoch', epoch)
        melodyseqs,durseqs,chordseqs,lows,highs,spseq = get_sequences(NPATH,DPATH,CPATH,PPATH,SPPATH)
        latest_g_loss,latest_d_loss,actual_seq, actual_seq_dur, \
        sup_gen_x, sup_gen_x_dur, unsup_gen_x, unsup_gen_x_dur, \
        supervised_chord_keys, supervised_chord_keys_onehot, supervised_chord_notes, supervised_sps, \
        unsupervised_chord_keys, unsupervised_chord_keys_onehot, unsupervised_chord_notes, unsupervised_sps = train.train_epoch(
            sess, trainable_model, EPOCH_ITER,
            proportion_supervised=proportion_supervised,
            g_steps=G_STEPS, d_steps=D_STEPS,
            next_sequence=get_random_sequence,
            sequences=melodyseqs,durseqs=durseqs,chordseqs=chordseqs,lows=lows,highs=highs,spseq=spseq,
            verify_sequence=None,skipDiscriminator = skipD,skipGenerator = skipG, note_adjust=NOTEADJUST)
        actuals.append([actual_seq,actual_seq_dur,supervised_chord_notes,supervised_chord_keys,supervised_chord_keys_onehot, supervised_sps])
        sups.append([sup_gen_x,sup_gen_x_dur,supervised_chord_notes,supervised_chord_keys,supervised_chord_keys_onehot, supervised_sps])
        unsups.append([unsup_gen_x,unsup_gen_x_dur,unsupervised_chord_notes,unsupervised_chord_keys,unsupervised_chord_keys_onehot, unsupervised_sps])
        if not startUnsup and latest_d_loss != None and latest_d_loss < 0.5:
            print('###### FREEZING DISCRIMINATOR')
            skipD = True
        if latest_g_loss != None and latest_g_loss < 2.0:
            startUnsup = True
        if startUnsup:
            skipG = False
            skipD = False
            curric_count+=1
            proportion_supervised = max(SUP_BASELINE, 0.0 - CURRICULUM_RATE * curric_count)
            #if latest_d_loss != None and latest_d_loss < 0.3:
                #print('###### FREEZING DISCRIMINATOR')
                #skipD = True
                #proportion_supervised = 0.1
                #startUnsup = False

        seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,start_pitch = get_random_sequence(melodyseqs,durseqs,chordseqs,lows,highs,spseq)
        #keyshift = random.randint(1,7)
        #for i in range(len(chordkeys)):
            #chordkeys[i] = (chordkeys[i]+keyshift) % 12
        gen_x, gen_x_dur = trainable_model.generate(sess,chordkeys,chordkeys_onehot,chordnotes,sequence_length,start_pitch)
        gen_x = [x for x in gen_x]
        gen_x_dur = [x for x in gen_x_dur]
        generations.append([gen_x,gen_x_dur,chordnotes,chordkeys,chordkeys_onehot,start_pitch])

    all_seqs = [actuals, sups, unsups]
    for seqs in all_seqs:
        for seq in seqs:
            if seq != None:
                for k in range(len(seq)):
                    val = seq[k]
                    if val != None:
                        if isinstance(val, (list,tuple)):
                            for i in range(len(val)):
                                if isinstance(val[i],(list,tuple)):
                                    for j in range(len(val[i])):
                                        val[i][j] = int(val[i][j])
                                elif val[i] != None:
                                    val[i] = int(val[i])
                        else:
                            seq[k] = int(seq[k])
    with open("generations4.json",'w') as dumpfile:
        json.dump(all_seqs, dumpfile)

    for seq in generations:
        if seq != None:
            for k in range(len(seq)):
                val = seq[k]
                if val != None:
                    if isinstance(val, (list,tuple)):
                        for i in range(len(val)):
                            if isinstance(val[i],(list,tuple)):
                                for j in range(len(val[i])):
                                    val[i][j] = int(val[i][j])
                            elif val[i] != None:
                                val[i] = int(val[i])
                    else:
                        seq[k] = int(seq[k])
    with open("generations_g.json",'w') as dumpfile:
        json.dump(generations,dumpfile)

if __name__ == '__main__':
    test_sequence_definition()
    main()
