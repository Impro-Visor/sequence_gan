from __future__ import print_function

__doc__ = """Simple demo on ii-V-I leadsheets.

"""

import model_timestep
import train_timestep
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
    parsename = "leadsheets_bricked_all2"
    parsename2 = "transcriptions"

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

ALLPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_features.json"
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
NPATH2 = "./parsed_"+parsename2+dur_append+"/"+expertname+"expert_"+encodingname+"_melodies.json"
DPATH2 = "./parsed_"+parsename2+dur_append+"/"+expertname+"expert_"+encodingname+"_durs.json"
CPATH2 = "./parsed_"+parsename2+dur_append+"/"+expertname+"expert_"+encodingname+"_chords.json"
PPATH2 = "./parsed_"+parsename2+dur_append+"/"+expertname+"expert_"+encodingname+"_pos.json"
SPPATH2 = "./parsed_"+parsename2+dur_append+"/"+expertname+"expert_"+encodingname+"_startpitches.json"
print(NPATH2)
print(DPATH2)
print(CPATH2)
print(PPATH2)
print(SPPATH2)

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
    MIDI_MIN = 36#55#53#46#44#55
    MIDI_MAX = 108#82#89#96#106#84
REST_VAL = MIDI_MAX-MIDI_MIN+1
NUM_EMB = MIDI_MAX-MIDI_MIN+2
NUM_EMB_DUR = 2#48#15       using 2 because attack
NUM_EMB_DURA = 2#48         using 2 because attack
NUM_EMB_CHORD = 12
EMB_DIM = 72
EMB_DIM_DUR = 2#96
EMB_DIM_DURA = 2#96
EMB_DIM_CHORD = 24
HIDDEN_DIM = 300 # 100 nodes is the min.
HIDDEN_DIM_B = 50
NUMBER_HIDDEN_LAYERS = 1
MAX_SEQ_LENGTH = 48*4
MAX_SEQ_DUR_LENGTH = 48*4
EXPANSION_FACTOR = 1
MIN_BLOCK_LENGTH = 1
MAX_BLOCK_LENGTH = 16
END_COUNT = 48
if LEADSHEET_CHOICE == TWOFIVEONE:
    MAX_SEQ_LENGTH = 20
    MAX_BLOCK_LENGTH = 2
    MIN_BLOCK_LENGTH = 2
elif LEADSHEET_CHOICE == TRANSCRIPTIONS:
    MAX_SEQ_LENGTH = 48
    MAX_SEQ_DUR_LENGTH = 48
    EXPANSION_FACTOR = 16 # Multiply by 16 for extraneous generations
    MIN_BLOCK_LENGTH = 2
    MAX_BLOCK_LENGTH = 2

EPOCH_ITER = 100
SWITCH_EPOCH = -1
SAVE_EPOCH = 10
SAVE_INTERVAL = 10
RESET_INTERVAL = 500
INIT_PROPORTION_SUPERVISED = 1.0
CURRICULUM_RATE = 0.1  # how quickly to move from supervised training to unsupervised
SUP_BASELINE = 0.0 # Decrease ratio of supervised training to this baseline ratio.
TRAIN_ITER = EPOCH_ITER * 100000  # generator/discriminator alternating
G_STEPS = 7  # how many times to train the generator each round
D_STEPS = 1  # how many times to train the discriminator per generator steps
G_LOSS_BOUNDARY = 1.5 # how far the supervised trainer must reach
LEARNING_RATE = 1e-3 #* MAX_SEQ_LENGTH # 1e-3 is stable-ish, 1e-2 oscillates, 1e-4 is slow
SEED = 88

def get_trainable_model():
    return model_timestep.GRU(
        NUM_EMB, NUM_EMB_DUR, NUM_EMB_DURA, NUM_EMB_CHORD,
        EMB_DIM, EMB_DIM_DUR, EMB_DIM_DURA, EMB_DIM_CHORD,
        HIDDEN_DIM, HIDDEN_DIM_B, NUMBER_HIDDEN_LAYERS,
        MAX_SEQ_DUR_LENGTH, MAX_BLOCK_LENGTH,END_COUNT,
        learning_rate=LEARNING_RATE,MIDI_MIN=MIDI_MIN,MIDI_MAX=MIDI_MAX,ENCODING=ENCODING)
"""

    return {"full_chords":clist, 
            "pitches_skip":mlist,
            "pos_skip":poslist,
            "chord_keys":ckeylist,
            "namelist":namelist,
            "durs":dlist,
            "starts":splist,
            "attacks":alist,
            "pitches_noskip":mlist_noskip,
            "pos_noskip":poslist_noskip,
            "transposed_seqs_skip":transposed_seqs_skip, # 0th index: leadsheet, 1st index: transpose, 2nd index: sequence, 3rd index: item at notecount
            "transposed_seqs_noskip":transposed_seqs_noskip}
"""
def get_sequences_onefile_timestep(allpath):
    print(allpath)
    with open(allpath,'r') as infile:
        data = json.load(infile)
    mt_list = data["transposed_seqs_noskip"] # 0th index: leadsheet, 1st index: transpose, 2nd index: sequence, 3rd index: item at timecount
    ct_list = data["full_chords"] # 0th index: leadsheet, 1st index: transpose, 2nd index: item at timecount
    """clists = []
    for seq in mt_list[0][0][:3]:
        print(seq)
        clists.append(ct_list[0][seq[0][-1]:seq[-1][-1]])
    print([x[0] for x in clists[0]])
    print("blah")
    mt_lista = np.array(mt_list)
    print(mt_lista.shape)
    print(mt_list[0][0][0])
    ct_lista = np.array(ct_list)
    print(ct_lista.shape)
    print(ct_list[0][0][0])
    print(np.array(data["transposed_seqs_skip"]).shape)
    print(data["transposed_seqs_skip"][0][0][0])
    print("blah2")
    #print(mt_list[0][0][:10])
    #print(ct_list[5][0][0])"""
    all_seqs = []
    maxseqdur = 0
    for lindex in range(len(mt_list)): # For each leadsheet
        mt = mt_list[lindex]
        ct = ct_list[lindex]
        for tindex in range(len(mt)): # For each transpose section
            m = mt[tindex]
            c = ct[tindex]
            for seq in m: # For each brick sequence

                # Clear out starting rests
                seqStart = -1
                for i in range(len(seq)):
                    if seq[i][0] != REST_VAL:
                        seqStart = i
                        break
                if seqStart == -1:
                    continue

                # Clear out ending rests
                seqEnd = -1
                for i in range(len(seq)):
                    if seq[len(seq)-1-i][0] != REST_VAL:
                        seqEnd = len(seq)-i
                        break
                if seqEnd != -1:
                    seq = seq[seqStart:seqEnd]
                else:
                    seq = seq[seqStart:]

                # Get primer of 4 timesteps
                startseqsize = 4
                if len(seq) <= startseqsize or len(seq) > MAX_SEQ_LENGTH:
                    continue
                startseq = seq[:startseqsize]
                startseq.reverse()
                spseq = []# noteval, dur, beatpos, chordkey, otherdur
                for pitch,attack,index in startseq:
                    spseq.append([pitch,attack,attack, c[index][0],attack])

                # Get rest of sequence
                seq = seq[startseqsize:]
                noteseq = []
                durseq = []
                chordseq = []
                lowseq = []
                highseq = []
                seqdur = 0
                for pitch,attack,index in seq:
                    noteseq.append(pitch)
                    durseq.append(attack)
                    chordseq.append(c[index])
                    seqdur+=1
                    if pitch == REST_VAL:
                        lowseq.append(0.0)
                        highseq.append(0.0)
                    else:
                        lowseq.append(float(pitch/float(MIDI_MAX-MIDI_MIN)))
                        highseq.append(1.0-float(pitch/float(MIDI_MAX-MIDI_MIN)))
                if seqdur > MAX_SEQ_DUR_LENGTH:
                    continue
                all_seqs.append([noteseq,durseq,chordseq,lowseq,highseq,spseq])
                if seqdur > maxseqdur:
                    maxseqdur=seqdur

    print(maxseqdur)
    all_seqs.sort(key=lambda x: len(x[0]),reverse=False) # Sort by sequence length
    noteseqs = [x[0] for x in all_seqs]
    durseqs = [x[1] for x in all_seqs]
    chordseqs = [x[2] for x in all_seqs]
    lows = [x[3] for x in all_seqs]
    highs = [x[4] for x in all_seqs]
    spseqs = [x[5] for x in all_seqs]
    print("Number of sequences: ", len(noteseqs))
    return noteseqs,durseqs,chordseqs,lows,highs,spseqs

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
    for nthing in spseq:
        for blah in nthing:
            for elem in blah:
                assert elem >= 0
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
    chordkeys_onehot = [x[0] for x in chordseq]
    chordkeys = [x[0] for x in chordseq]
    chordnotes = [x[1] for x in chordseq]
    #print(chordnotes)
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
    generations5 = []
    generations7 = []
    curric_count = 0
    proportion_supervised = INIT_PROPORTION_SUPERVISED
    skipD = False
    skipG = False
    startUnsup = False
    ii = 0
    direction = 1
    melodyseqs,durseqs,chordseqs,lows,highs,spseq = get_sequences_onefile_timestep(ALLPATH)
    saveNum = 0
    for epoch in range(TRAIN_ITER // EPOCH_ITER):
        print('epoch', epoch)
        ii,direction,latest_g_loss,latest_d_loss,actual_seq, actual_seq_dur, \
        sup_gen_x, sup_gen_x_dur, unsup_gen_x, unsup_gen_x_dur, \
        supervised_chord_keys, supervised_chord_keys_onehot, supervised_chord_notes, supn0,supn1,supn2,supn3, \
        unsupervised_chord_keys, unsupervised_chord_keys_onehot, unsupervised_chord_notes, unsupn0,unsupn1,unsupn2,unsupn3 = train_timestep.train_epoch(
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
        if (latest_g_loss != None and latest_g_loss < G_LOSS_BOUNDARY) or epoch > 1000:
            startUnsup = True
        if startUnsup:
            skipG = False
            skipD = False
            curric_count+=1
            proportion_supervised = max(SUP_BASELINE, 0.0 - CURRICULUM_RATE * curric_count)

        # Normal generations
        ii,direction,seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3 = get_random_sequence(ii,direction,melodyseqs,durseqs,chordseqs,lows,highs,spseq)
        lengths = get_random_sequence_lengths(sequence_length)
        gen_x, gen_x_dur = trainable_model.generate(sess,lengths,chordkeys,chordkeys_onehot,chordnotes,sequence_length,n0,n1,n2,n3)
        gen_x = [x for x in gen_x]
        gen_x_dur = [x for x in gen_x_dur]
        generations.append([gen_x,gen_x_dur,chordnotes,chordkeys,chordkeys_onehot,n0,n1,n2,n3])

        # Off-beat starts +5
        ii,direction,seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3 = get_random_sequence(ii,direction,melodyseqs,durseqs,chordseqs,lows,highs,spseq)
        #n0[0] = (n0[1]+5) % 48
        #n1[0] = (n1[1]+5) % 48
        #n2[0] = (n2[1]+5) % 48
        #n3[1] = (n3[1]+5) % 48
        startindex=-1
        for j in range(len(seq)):
            if seq_dur[j] == 0:
                startindex=j
                break
        if startindex != -1:
            for _ in range(5):
                seq = [seq[startindex]] + seq
                seq_dur = [seq_dur[startindex]] + seq_dur
                chordkeys = [chordkeys[startindex]] + chordkeys
                chordkeys_onehot = [chordkeys_onehot[startindex]] + chordkeys_onehot
                chordnotes = [chordnotes[startindex]] + chordnotes
                low = [low[startindex]] + low
                high = [high[startindex]] + high
                sequence_length += 1

        lengths = get_random_sequence_lengths(sequence_length)
        gen_x, gen_x_dur = trainable_model.generate(sess,lengths,chordkeys,chordkeys_onehot,chordnotes,sequence_length,n0,n1,n2,n3)
        gen_x = [x for x in gen_x]
        gen_x_dur = [x for x in gen_x_dur]
        generations5.append([gen_x,gen_x_dur,chordnotes,chordkeys,chordkeys_onehot,n0,n1,n2,n3])

        # Off-beat starts +7
        ii,direction,seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3 = get_random_sequence(ii,direction,melodyseqs,durseqs,chordseqs,lows,highs,spseq)
        #n0[0] = (n0[1]+7) % 48
        #n1[0] = (n1[1]+7) % 48
        #n2[0] = (n2[1]+7) % 48
        #n3[1] = (n3[1]+7) % 48
        startindex=-1
        for j in range(len(seq)):
            if seq_dur[j] == 0:
                startindex=j
                break
        if startindex != -1:
            for _ in range(5):
                seq = [seq[startindex]] + seq
                seq_dur = [seq_dur[startindex]] + seq_dur
                chordkeys = [chordkeys[startindex]] + chordkeys
                chordkeys_onehot = [chordkeys_onehot[startindex]] + chordkeys_onehot
                chordnotes = [chordnotes[startindex]] + chordnotes
                low = [low[startindex]] + low
                high = [high[startindex]] + high
                sequence_length += 1

        lengths = get_random_sequence_lengths(sequence_length)
        gen_x, gen_x_dur = trainable_model.generate(sess,lengths,chordkeys,chordkeys_onehot,chordnotes,sequence_length,n0,n1,n2,n3)
        gen_x = [x for x in gen_x]
        gen_x_dur = [x for x in gen_x_dur]
        generations7.append([gen_x,gen_x_dur,chordnotes,chordkeys,chordkeys_onehot,n0,n1,n2,n3])

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
            with open("tgens_timestep_t"+str(ecount)+".json",'w') as dumpfile:
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
            for seq in generations5:
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
            for seq in generations7:
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
            with open("tgens_timestep_g"+str(ecount)+".json",'w') as dumpfile:
                json.dump(generations,dumpfile)
            with open("tgens_timestep_5g"+str(ecount)+".json",'w') as dumpfile:
                json.dump(generations5,dumpfile)
            with open("tgens_timestep_7g"+str(ecount)+".json",'w') as dumpfile:
                json.dump(generations7,dumpfile)

            print("------SAVING CURRENT MODEL------")
            savefile = 'savedir_tnew/'+parsename+'_model' + str(saveNum)
            saver = tf.train.Saver()
            saver.save(sess, savefile)
            saver.export_meta_graph(savefile+'.meta')
            saveNum+=1

        if epoch % RESET_INTERVAL == 0:
            actuals = []
            sups = []
            unsups = []
            generations = []
        if epoch == SWITCH_EPOCH:
            print("------SWITCHING DATASETS------")
            print(NPATH2)
            print(DPATH2)
            print(CPATH2)
            print(PPATH2)
            print(SPPATH2)
            actuals = []
            sups = []
            unsups = []
            generations = []
            ii = 0
            direction = 1
            melodyseqs,durseqs,chordseqs,lows,highs,spseq = get_sequences_onefile(ALLPATH)
if __name__ == '__main__':
    main()
