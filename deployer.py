import json
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import copy

GENERATING = True
DISCRIMINATING_REAL = False
DISCRIMINATING_PRIMED = False
PRIMED = False

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

NPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_melodies.json"
DPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_durs.json"
CPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_chords.json"
PPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_pos.json"
SPPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_startpitches.json"
ALLPATH = "./parsed_"+parsename+dur_append+"/"+expertname+"expert_"+encodingname+"_features.json"
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
    MIDI_MIN = 36#45#55#53#46#44#55
    MIDI_MAX = 108#94#82#89#96#106#84
    NUM_EMB = MIDI_MAX-MIDI_MIN+2

if LEADSHEET_CHOICE == TWOFIVEONE:
    MAX_SEQ_LENGTH = 20
    MAX_BLOCK_LENGTH = 2
    MIN_BLOCK_LENGTH = 2
elif LEADSHEET_CHOICE == TRANSCRIPTIONS:
    MAX_SEQ_LENGTH = 48
    MIN_BLOCK_LENGTH = 2
    MAX_BLOCK_LENGTH = 2
SEED = 88
REST_PITCH = MIDI_MAX+1-MIDI_MIN
REST_VAL = REST_PITCH

NUM_GENS = 20000
NUM_DISC = 1000
pitches_p = [73,78,80,81]
pitches_p = [x-MIDI_MIN for x in pitches_p]
durs_p = [6,6,6,6]
start_p = 0
chordkeys_p = [7,7,7,7]

basedir = '/home/nic/sequence_gan/savedir_new_beatpos/'
modelmeta_filename = basedir+"leadsheets_bricked_all2_model199.meta"
modeldir_filename = basedir
dump_filename = "deployer_gens_beatpos.json"

realseqdump_filename = "deployer_realseqs0.json"
probdump_filename = "deployer_probs0.json"
dummyseqdump_filename = "deployer_dummyseqs0.json"
dummyprobdump_filename = "deployer_dummyprobs0.json"

dummy_pitches = [REST_PITCH]*12
dummy_pitches = [x-MIDI_MIN for x in dummy_pitches]
dummy_durs = [6]*12
dummy_start = 0
dummy_ck = [0]*12
dummy_ckn = [[1,0,0,0,1,0,0,1,0,0,0,0]]*12

def convertDummyToSeq(pitches,durs,startpos,chordkeys,chordkey_notes):
    lows = []
    highs = []
    for pitch in pitches:
        isRest = (pitch == REST_PITCH)
        low = 0.0 if isRest else float(pitch)/float(MIDI_MAX-MIDI_MIN)
        high = 0.0 if isRest else 1-low
        lows.append(low)
        highs.append(high)
    spseq = []
    for i in range(4):
        spnote = [pitches[i],(startpos + durs[i]) % 48, startpos % 48, chordkeys[i], (durs[i]-1+48)%48]
        startpos += durs[i]
        spseq.append(spnote)
    length = len(pitches)
    lengths = get_random_sequence_lengths(length)
    return length,lengths,pitches,durs,chordkeys,chordkeys,chordkey_notes,lows,highs,spseq[0],spseq[1],spseq[2],spseq[3]

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

def get_random_sequence(sequences,durseqs,chordseqs,lows,highs,spseq):
    """
    Get a random note sequence from training set.
    """
    i = np.random.randint(len(sequences))
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
    return notes,durs,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3

def generate_with_primer(chordseq,pitches,durs,startpos,chordkeys):
    length = len(chordseq)
    spseq = []
    for i in range(4):
        spnote = [pitches[i],(startpos + durs[i]) % 48, startpos % 48, chordkeys[i], (durs[i]-1+48)%48]
        startpos += durs[i]
        spseq.append(spnote)
    return chordseq,spseq,len(chordseq)


def decompose_length(sequence_length):
    while sequence_length > 0:
        length = min(random.randint(MIN_BLOCK_LENGTH,MAX_BLOCK_LENGTH),sequence_length)
        yield length
        sequence_length -= length

def generate_random_sequence(chordseq):
    length = len(chordseq)
    firstchord = chordseq[0]
    noteseq = np.random.randint(0,high=NUM_EMB,size=4)
    durseq = np.random.randint(0,high=48,size=4)+1
    index_count = 0
    spseq = []
    for i in range(4):
        spnote = [noteseq[i],(index_count + durseq[i]) % 48,index_count % 48,firstchord[0],(durseq[i]-1+48) % 48]
        index_count += durseq[i]
        spseq.append(spnote)

    return chordseq,spseq,length

def decompose_random_sequence(chordseq,spseq):
    """
    Get a random note sequence from training set.
    """
    chordkeys = np.array([x[0] for x in chordseq])
    chordkeys_onehot = [x[0] for x in chordseq]
    chordkeys = [x[0] for x in chordseq]
    chordnotes = [x[1] for x in chordseq]
    n0 = spseq[0]
    n1 = spseq[1]
    n2 = spseq[2]
    n3 = spseq[3]
    return chordkeys,chordkeys_onehot,chordnotes,n3,n2,n1,n0

def get_random_sequence_lengths(sequence_length):
    return list(decompose_length(sequence_length))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x))

def sigmoid(x):
    """Compute sigmoid values for each sets of scores in x."""
    return 1 / (1 + np.exp(-x))

def sigmoid_cross_entropy(x):
    return -x+np.log(1+np.exp(x))

def discriminate(session, lengths,x, x_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3):
    inputs = [lengths,x, x_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3]
    #for i in inputs:
    #    print(i)
    start_dura = n0[4]
    start_chordkey = n0[3]
    p0 = n0[0]
    p1 = n1[0]
    p2 = n2[0]
    p3 = n3[0]
    d0 = n0[1]
    d1 = n1[1]
    d2 = n2[1]
    d3 = n3[1]
    output = session.run(["d_real_predictions_out:0"],
                          feed_dict={"x:0": x, "x_dur:0": x_dur,
                                     "lows:0": low, "highs:0": high, 
                                     "lengths:0": lengths,
                                     "chordKeys:0": chordkeys, 
                                     "chordKeys_onehot:0": chordkeys_onehot, 
                                     "chordNotes:0": chordnotes,
                                     "sequence_length:0": sequence_length,
                                     "p0:0":p0,"p1:0":p1,"p2:0":p2,"p3:0":p3,
                                     "d0:0":d0,"d1:0":d1,"d2:0":d2,"d3:0":d3, 
                                     "start_chordkey:0":start_chordkey, "start_dura:0":start_dura})
    return output[0]

def generate(session,lengths,chordkeys,chordkeys_onehot,chordnotes,sequence_length,n0,n1,n2,n3):
    #print(len(chordkeys_onehot))
    #inputs = [lengths,chordkeys,chordkeys_onehot,chordnotes,sequence_length,n3,n2,n1,n0]
    #for i in inputs:
    #    print(i)
    start_dura = n0[4]
    start_chordkey = n0[3]
    p0 = n0[0]
    p1 = n1[0]
    p2 = n2[0]
    p3 = n3[0]
    d0 = n0[1]
    d1 = n1[1]
    d2 = n2[1]
    d3 = n3[1]
    outputs = session.run(
            ["gen_x_out:0", "gen_x_dur_out:0"],
            feed_dict={"lengths:0": lengths,
                       "samples:0": np.random.uniform(size=len(chordkeys_onehot)),
                       "samples_dur:0": np.random.uniform(size=len(chordkeys_onehot)),
                       "chordKeys:0": chordkeys, 
                       "chordKeys_onehot:0": chordkeys_onehot, 
                       "chordNotes:0": chordnotes,
                       "sequence_length:0": sequence_length,
                       "p0:0":p0,"p1:0":p1,"p2:0":p2,"p3:0":p3,
                       "d0:0":d0,"d1:0":d1,"d2:0":d2,"d3:0":d3, 
                       "start_chordkey:0":start_chordkey, "start_dura:0":start_dura})
    return outputs

def discriminate_primed():
    print("DISCRIMINATING PRIMED SEQS")
    with tf.Session() as sess:

        print("Loading model...")
        saver = tf.train.import_meta_graph(modelmeta_filename)
        saver.restore(sess, tf.train.latest_checkpoint(modeldir_filename))
        print("Loaded model.")
        print("Evaluating dummy sequence...")
        sequence_length,lengths,seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,n0,n1,n2,n3 = convertDummyToSeq(dummy_pitches,dummy_durs,dummy_start,dummy_ck,dummy_ckn)
        probabilities = discriminate(sess,lengths,seq,seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3)
        print(sigmoid(probabilities))
        probabilities = probabilities.tolist()
        print("Evaluted dummy sequence.")
        dummy_seq_forsave = [seq,seq_dur,chordnotes,chordkeys,chordkeys_onehot,n0,n1,n2,n3]
        for k in range(len(dummy_seq_forsave)):
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

        print("Dumping...")
        with open(dummyseqdump_filename,'w') as dumpfile:
            json.dump(dummy_seq_forsave,dumpfile)
        with open(dummyprobdump_filename,'w') as dumpfile:
            json.dump(probabilities,dumpfile)
        print("Dumped.")

def discriminate_real():
    print("DISCRIMINATING REAL SEQS")
    with tf.Session() as sess:

        print("Loading model...")
        saver = tf.train.import_meta_graph(modelmeta_filename)
        saver.restore(sess, tf.train.latest_checkpoint(modeldir_filename))
        print("Loaded model.")

        print("Getting real sequences...")
        melodyseqs,durseqs,chordseqs,lows,highs,spseq = get_sequences(NPATH,DPATH,CPATH,PPATH,SPPATH)
        print("Got real sequences.")
        last_probs = []
        prob_list = []
        seq_list = []
        print("Evaluating sequences...")
        for nd in range(NUM_DISC):
            seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3 = get_random_sequence(melodyseqs,durseqs,chordseqs,lows,highs,spseq)
            lengths = get_random_sequence_lengths(sequence_length)
            probabilities = discriminate(sess,lengths,seq,seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3)
            #print("Sequence " + str(nd) + " probabilities:")
            #print(softmax(probabilities).tolist())
            sig_probs = sigmoid(probabilities).tolist()
            #print(sig_probs)
            last_prob = sigmoid(probabilities).tolist()[-1]
            #print(last_prob)
            last_probs.append(last_prob)
            probabilities = probabilities.tolist()
            prob_list.append(probabilities)
            seq_list.append([seq,seq_dur,chordnotes,chordkeys,chordkeys_onehot,n0,n1,n2,n3])
        print("Evaluated sequences.")
        plt.hist(np.array(last_probs),range=(0.0,1.0),bins=500)

        plt.savefig("probs_real3.png")
        for seq in seq_list:
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
        print("Dumping...")
        with open(realseqdump_filename,'w') as dumpfile:
            json.dump(seq_list,dumpfile)
        with open(probdump_filename,'w') as dumpfile:
            json.dump(prob_list,dumpfile)
        print("Dumped.")

def get_sequences_beatposition(allpath):
    MAX_SEQ_DUR_LENGTH = 48*4
    print(allpath)
    with open(allpath,'r') as infile:
        data = json.load(infile)
    mt_list = data["transposed_seqs_skip"]
    ct_list = data["full_chords"]
    #print(ct_list[5][0][0])
    all_seqs = []
    maxseqdur = 0
    for lindex in range(len(mt_list)):
        mt = mt_list[lindex]
        ct = ct_list[lindex]
        for tindex in range(len(mt)):
            m = mt[tindex]
            c = ct[tindex]
            for seq in m:
                seqStart = -1
                for i in range(len(seq)):
                    if seq[i][0] != REST_VAL:
                        seqStart = i
                        break
                if seqStart == -1:
                    continue
                seqEnd = -1
                for i in range(len(seq)):
                    if seq[len(seq)-1-i][0] != REST_VAL:
                        seqEnd = len(seq)-i
                        break
                if seqEnd != -1:
                    seq = seq[seqStart:seqEnd]
                else:
                    seq = seq[seqStart:]
                startseqsize = 4
                if len(seq) <= startseqsize:# or len(seq) > MAX_SEQ_LENGTH:
                    continue
                startseq = seq[:startseqsize]
                startseq.reverse()
                seq = seq[startseqsize:]
                noteseq = []
                durseq = []
                chordseq = []
                lowseq = []
                highseq = []
                prevpos = 0
                spseq = []# noteval, dur, beatpos, chordkey, otherdur
                for pitch,beatpos,index in startseq:
                    dur = (beatpos-prevpos+48-1)%48 # actual durs are 1-48, but one-hotted is 0-47
                    spseq.append([pitch,beatpos,dur, c[index][0],dur])
                    prevpos = beatpos
                    #for _ in range(dur):
                    #    chordseq.append(c[index])
                seqdur = 0
                for pitch,beatpos,index in seq:
                    noteseq.append(pitch)
                    durseq.append(beatpos)
                    dur = (beatpos-prevpos+48-1)%48+1 # -1 +1 for octaves
                    prevpos = beatpos
                    seqdur+=dur
                    for _ in range(dur):
                        chordseq.append(c[index])
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
                    prevpos = 0
                    durs = []
                    for pitch,beatpos,index in seq:
                        durs.append((beatpos-prevpos+48-1)%48+1)
                        prevpos = beatpos
                    print(durs)
                    print(sum(durs))

    print(maxseqdur)
    all_seqs.sort(key=lambda x: len(x[0]),reverse=False)
    noteseqs = [x[0] for x in all_seqs]
    durseqs = [x[1] for x in all_seqs]
    chordseqs = [x[2] for x in all_seqs]
    lows = [x[3] for x in all_seqs]
    highs = [x[4] for x in all_seqs]
    spseqs = [x[5] for x in all_seqs]
    print("Number of sequences: ", len(noteseqs))
    return noteseqs,durseqs,chordseqs,lows,highs,spseqs


def get_random_sequence_i(i,direction,sequences,durseqs,chordseqs,lows,highs,spseq):
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
    n0 = copy.deepcopy(spseq[i][0])
    n1 = copy.deepcopy(spseq[i][1])
    n2 = copy.deepcopy(spseq[i][2])
    n3 = copy.deepcopy(spseq[i][3])
    
    # start_pitch = spseq[i][0]
    # start_duration = spseq[i][1]
    # start_beat = spseq[i][2]
    # start_chordkey = spseq[i][3]
    # start_dura = spseq[i][4]
    return i,direction,notes,durs,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3

def generate_master():
    print("GENERATING REAL SEQS")
    generations = []
    ONLY_CHORDS = False
    DO_PROBS = False

    with tf.Session() as sess:
        # LOAD SAVED MODEL
        print("Loading model...")
        saver = tf.train.import_meta_graph(modelmeta_filename)
        saver.restore(sess, tf.train.latest_checkpoint(modeldir_filename))
        print("Loaded model.")
        
        # LOAD CHORDS
        with open(ALLPATH,'r') as featurefile:
            noteseqs,durseqs,chordseqs,lows,highs,spseqs = get_sequences_beatposition(ALLPATH)
        
        print("Generating...")
        counttracker = NUM_GENS/10
        for i in range(NUM_GENS):
            if i % counttracker == 0:
                print("Gen:",i)
            if ONLY_CHORDS:
                if PRIMED:
                    chordseq,spseq,sequence_length = generate_with_primer(chordseqs[np.random.randint(0,numseqs)],pitches_p,durs_p,start_p,chordkeys_p)
                else:
                    chordseq,spseq,sequence_length = generate_random_sequence(chordseqs[np.random.randint(0,numseqs)])
                lengths = get_random_sequence_lengths(sequence_length)
                chordkeys,chordkeys_onehot,chordnotes,n0,n1,n2,n3 = decompose_random_sequence(chordseq,spseq)
            else:
                seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3 = get_random_sequence(noteseqs,durseqs,chordseqs,lows,highs,spseqs)
                lengths = get_random_sequence_lengths(sequence_length)
                assert(len(chordkeys_onehot) > 1)

            #print("Generating...")
            gen_x, gen_x_dur = generate(sess,lengths,chordkeys,chordkeys_onehot,chordnotes,sequence_length,n0,n1,n2,n3)
            #print("Generated.")

            gen_x = [x for x in gen_x]
            # if len(gen_x) > 1:
            #     print(gen_x)
            gen_x_dur = [x for x in gen_x_dur]

            generations.append([gen_x,gen_x_dur,chordnotes,chordkeys,chordkeys_onehot,n0,n1,n2,n3])
        print("Generated.")

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
        
        print("Dumping...")
        with open(dump_filename,'w') as dumpfile:
            json.dump(generations,dumpfile)
        print("Dumped.")

        if DO_PROBS:
            print("DISCRIMINATING GEN SEQS")
            """with tf.Session() as sess2:

            print("Loading model...")
            saver = tf.train.import_meta_graph(modelmeta_filename)
            saver.restore(sess2, tf.train.latest_checkpoint(modeldir_filename))
            print("Loaded model.")"""

            last_probs = []
            prob_list = []
            seq_list = []
            print("Evaluating sequences...")
            for (seq,seq_dur,chordnotes,chordkeys,chordkeys_onehot,n0,n1,n2,n3) in generations:
                sequence_length = len(seq)
                lengths = get_random_sequence_lengths(sequence_length)

                lows = []
                highs = []
                for pitch in seq:
                    isRest = (pitch == REST_PITCH)
                    low = 0.0 if isRest else float(pitch)/float(MIDI_MAX-MIDI_MIN)
                    high = 0.0 if isRest else 1-low
                    lows.append(low)
                    highs.append(high)
                probabilities = discriminate(sess,lengths,seq,seq_dur,chordkeys,chordkeys_onehot,chordnotes,lows,highs,sequence_length,n0,n1,n2,n3)
                #print("Sequence " + str(nd) + " probabilities:")
                #print(softmax(probabilities).tolist())
                sig_probs = sigmoid(probabilities).tolist()
                #print(sig_probs)
                last_prob = np.mean(sigmoid(probabilities))
                #print(last_prob)
                last_probs.append(last_prob)
                probabilities = probabilities.tolist()
                prob_list.append(probabilities)
                seq_list.append([seq,seq_dur,chordnotes,chordkeys,chordkeys_onehot,n0,n1,n2,n3])
            print("Evaluated sequences.")
            plt.hist(np.array(last_probs),range=(0.0,1.0),bins=500)

            plt.savefig("probs_gen_restprimed.png")

def main():
    if DISCRIMINATING_PRIMED:
        discriminate_primed()

    if DISCRIMINATING_REAL:
        discriminate_real()

    if GENERATING:
        generate_master()

if __name__ == '__main__':
    main()
