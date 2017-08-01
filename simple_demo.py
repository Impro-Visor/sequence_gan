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
LEADSHEET_CHOICE = TRANSCRIPTIONS

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
    parsename = "majors"

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
if EXPERT == PITCH:
    MIDI_MIN = 55 # lowest note value found in trainingset
    MIDI_MAX = 89 # highest note value found in trainingset
elif EXPERT == INTERVAL:
    NUM_EMB = 29 # Min int = -13, Max int = 14. 0-27 for jumps, 28 for rest.
    MIDI_MIN = -13 # lowest interval value found in trainingset
    MIDI_MAX = 14 # highest interval value found in trainingset
if LEADSHEET_CHOICE == TRANSCRIPTIONS:
    MIDI_MIN = 53#46#44#55
    MIDI_MAX = 89#96#106#84
NUM_EMB = MIDI_MAX-MIDI_MIN+2
NUM_EMB_DUR = 48#15
NUM_EMB_DURA = 48
NUM_EMB_CHORD = 12
EMB_DIM = 72
EMB_DIM_DUR = 96
EMB_DIM_DURA = 96
EMB_DIM_CHORD = 24
HIDDEN_DIM = 300 # 100 nodes is the min.
HIDDEN_DIM_B = 50
NUMBER_HIDDEN_LAYERS = 1
MAX_SEQ_LENGTH = 96
MIN_BLOCK_LENGTH = 1
MAX_BLOCK_LENGTH = 16
if LEADSHEET_CHOICE == TWOFIVEONE:
    MAX_SEQ_LENGTH = 20
    MAX_BLOCK_LENGTH = 2
    MIN_BLOCK_LENGTH = 2
elif LEADSHEET_CHOICE == TRANSCRIPTIONS:
    MAX_SEQ_LENGTH = 20
    MIN_BLOCK_LENGTH = 2
    MAX_BLOCK_LENGTH = 2

EPOCH_ITER = 100
SAVE_INTERVAL = 50
RESET_INTERVAL = 500
INIT_PROPORTION_SUPERVISED = 1.0
CURRICULUM_RATE = 0.1  # how quickly to move from supervised training to unsupervised
SUP_BASELINE = 0.0 # Decrease ratio of supervised training to this baseline ratio.
TRAIN_ITER = EPOCH_ITER * 100000  # generator/discriminator alternating
G_STEPS = 7  # how many times to train the generator each round
D_STEPS = 1  # how many times to train the discriminator per generator steps
G_LOSS_BOUNDARY = 1.0 # how far the supervised trainer must reach
LEARNING_RATE = 1e-3 * MAX_SEQ_LENGTH # 1e-3 is stable-ish, 1e-2 oscillates, 1e-4 is slow
SEED = 88

def get_trainable_model():
    return model.GRU(
        NUM_EMB, NUM_EMB_DUR, NUM_EMB_DURA, NUM_EMB_CHORD,
        EMB_DIM, EMB_DIM_DUR, EMB_DIM_DURA, EMB_DIM_CHORD,
        HIDDEN_DIM, HIDDEN_DIM_B, NUMBER_HIDDEN_LAYERS,
        MAX_SEQ_LENGTH, MAX_BLOCK_LENGTH,
        learning_rate=LEARNING_RATE,MIDI_MIN=MIDI_MIN,MIDI_MAX=MIDI_MAX,ENCODING=ENCODING)

def get_sequences(notepath,durpath,chordpath,pospath,startppath):
    """
    Get the training set of sequences.
    """
    noteseqs = []
    with open(notepath,'r') as notefile:
        noteseqs = json.load(notefile)
        for i in range(len(noteseqs)):
            noteseqs[i] = noteseqs[i]
    durseqs = []
    with open(durpath,'r') as durfile:
        durseqs = json.load(durfile)
    chordseqs = []
    with open(chordpath,'r') as chordfile:
        chordseqs = json.load(chordfile)
    lows = []
    highs = []
    with open(pospath, 'r') as posfile:
        posseqs = json.load(posfile)
        for posseq in posseqs:
            low = [x[0] for x in posseq]
            high = [x[1] for x in posseq]
            lows.append(low)
            highs.append(high)
    spseq = []
    with open(startppath, 'r') as spfile:
        spseq = json.load(spfile)

    all_seqs = [[noteseqs[i], durseqs[i], chordseqs[i], lows[i], highs[i], spseq[i]] for i in range(len(noteseqs))]
    all_seqs.sort(key=lambda x: len(x[0]),reverse=False)
    noteseqs = [x[0] for x in all_seqs]
    durseqs = [x[1] for x in all_seqs]
    chordseqs = [x[2] for x in all_seqs]
    lows = [x[3] for x in all_seqs]
    highs = [x[4] for x in all_seqs]
    spseq = [x[5] for x in all_seqs]
    print("Number of sequences: ", len(noteseqs))
    return noteseqs,durseqs,chordseqs,lows,highs,spseq

def get_random_sequence(i,direction,sequences,durseqs,chordseqs,lows,highs,spseq):
    """
    Get a random note sequence from training set.
    """
    i += direction*1
    if i >= len(sequences)-1 or i <= 0:
        direction *= -1
    notes = sequences[i]
    durs = durseqs[i]
    chordseq = chordseqs[i]
    chordkeys = np.array([x[0] for x in chordseq])
    chordkeys_onehot = [x[0] for x in chordseq]
    chordkeys = [x[0] for x in chordseq]
    chordnotes = [x[1] for x in chordseq]
    low = lows[i]
    high = highs[i]
    sequence_length = len(notes)
    n0 = spseq[i][0]
    n1 = spseq[i][1]
    n2 = spseq[i][2]
    n3 = spseq[i][3]
    
    # start_pitch = spseq[i][0]
    # start_duration = spseq[i][1]
    # start_beat = spseq[i][2]
    # start_chordkey = spseq[i][3]
    # start_dura = spseq[i][4]
    return i,direction,notes,durs,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3

def get_random_sequence_lengths(sequence_length):
    return list(decompose_length(sequence_length))

def decompose_length(sequence_length):
    while sequence_length > 0:
        length = min(random.randint(MIN_BLOCK_LENGTH,MAX_BLOCK_LENGTH),sequence_length)
        yield length
        sequence_length -= length

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    trainable_model = get_trainable_model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print('training')
    actuals = []
    sups = []
    unsups = []
    generations = []
    curric_count = 0
    proportion_supervised = INIT_PROPORTION_SUPERVISED
    skipD = False
    skipG = False
    startUnsup = False
    ii = 0
    direction = 1
    melodyseqs,durseqs,chordseqs,lows,highs,spseq = get_sequences(NPATH,DPATH,CPATH,PPATH,SPPATH)
    for epoch in range(TRAIN_ITER // EPOCH_ITER):
        print('epoch', epoch)
        ii,direction,latest_g_loss,latest_d_loss,actual_seq, actual_seq_dur, \
        sup_gen_x, sup_gen_x_dur, unsup_gen_x, unsup_gen_x_dur, \
        supervised_chord_keys, supervised_chord_keys_onehot, supervised_chord_notes, supn0,supn1,supn2,supn3, \
        unsupervised_chord_keys, unsupervised_chord_keys_onehot, unsupervised_chord_notes, unsupn0,unsupn1,unsupn2,unsupn3 = train.train_epoch(
            sess, trainable_model, EPOCH_ITER,
            proportion_supervised=proportion_supervised,
            g_steps=G_STEPS, d_steps=D_STEPS,
            next_sequence=get_random_sequence,next_sequence_lengths=get_random_sequence_lengths,
            sequences=melodyseqs,durseqs=durseqs,chordseqs=chordseqs,lows=lows,highs=highs,spseq=spseq,
            skipDiscriminator = skipD,skipGenerator = skipG, note_adjust=NOTEADJUST, ii=ii,direction=direction)
        actuals.append([actual_seq,actual_seq_dur,supervised_chord_notes,supervised_chord_keys,supervised_chord_keys_onehot, supn0,supn1,supn2,supn3])
        sups.append([sup_gen_x,sup_gen_x_dur,supervised_chord_notes,supervised_chord_keys,supervised_chord_keys_onehot, supn0,supn1,supn2,supn3])
        unsups.append([unsup_gen_x,unsup_gen_x_dur,unsupervised_chord_notes,unsupervised_chord_keys,unsupervised_chord_keys_onehot, unsupn0,unsupn1,unsupn2,unsupn3])
        #if not startUnsup and latest_d_loss != None and latest_d_loss < 0.5:
        #    print('###### FREEZING DISCRIMINATOR')
        #    skipD = True
        if latest_g_loss != None and latest_g_loss < G_LOSS_BOUNDARY:
            startUnsup = True
        if startUnsup:
            skipG = False
            skipD = False
            curric_count+=1
            proportion_supervised = max(SUP_BASELINE, 0.0 - CURRICULUM_RATE * curric_count)

        ii,direction,seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3 = get_random_sequence(ii,direction,melodyseqs,durseqs,chordseqs,lows,highs,spseq)
        lengths = get_random_sequence_lengths(sequence_length)
        gen_x, gen_x_dur = trainable_model.generate(sess,lengths,chordkeys,chordkeys_onehot,chordnotes,sequence_length,n0,n1,n2,n3)
        gen_x = [x for x in gen_x]
        gen_x_dur = [x for x in gen_x_dur]
        generations.append([gen_x,gen_x_dur,chordnotes,chordkeys,chordkeys_onehot,n0,n1,n2,n3])

        if (epoch+1) % SAVE_INTERVAL == 0:
            ecount = int(epoch/SAVE_INTERVAL)
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
            with open("generations_t"+str(ecount)+".json",'w') as dumpfile:
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
            with open("generations_g"+str(ecount)+".json",'w') as dumpfile:
                json.dump(generations,dumpfile)
        if epoch % RESET_INTERVAL == 0:
            actuals = []
            sups = []
            unsups = []
            generations = []

if __name__ == '__main__':
    main()
