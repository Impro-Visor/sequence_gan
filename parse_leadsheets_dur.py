import leadsheet
import os
import json

BITS = True
ONE_HOT = False
bits_or_onehot = ONE_HOT

PURE_PITCH = True
PITCH_MIDI = 0
PITCH_INTERVAL = 1
PITCH_CHORD = 2
INTERVAL = True
CHORD = False
interval_or_chord = INTERVAL
PITCH_REP = PITCH_MIDI if PURE_PITCH else (PITCH_INTERVAL if interval_or_chord == INTERVAL else PITCH_CHORD)
ldir = "./ii-V-I_leadsheets/"
category = "pitchexpert_" if PITCH_REP == PITCH_MIDI else ("intervalexpert_" if PITCH_REP == PITCH_INTERVAL else "chordexpert_")
encoding = "bit_" if bits_or_onehot == BITS else "onehot_"
parsedir = "./parsed_ii-V-I_leadsheets_dur/" + category + encoding
cdir = parsedir + "chords.json"
mdir = parsedir + 'melodies.json'
posdir = parsedir + 'pos.json'
ckeydir = parsedir + "chordkeys.json"
namedir = parsedir + 'names.json'
ddir = parsedir + 'durs.json'
outputDirList = [cdir,mdir,posdir,ckeydir,namedir,ddir]

MIDI_MIN = 55 # lowest note value found in trainingset
MIDI_MAX = 89 # highest note value found in trainingset
NUM_NOTES = MIDI_MAX-MIDI_MIN+1 # number of distinct notes in trainingset
SEQ_LEN = 96

MIN_INTERV = -13
MAX_INTERV = 14

WHOLE = 0
WHOLE_DOTTED = 1
WHOLE_TRIPLET = 2
HALF = 3
HALF_DOTTED = 4
HALF_TRIPLET = 5
QUARTER = 6
QUARTER_DOTTED = 7
QUARTER_TRIPLET = 8
EIGHTH = 9
EIGHTH_DOTTED = 10
EIGHTH_TRIPLET = 11
SIXTEENTH = 12
SIXTEENTH_DOTTED = 13
SIXTEENTH_TRIPLET = 14

DURATION_MAPPING = {
    1: SIXTEENTH_TRIPLET,
    2: EIGHTH_TRIPLET,
    3: SIXTEENTH,
    4: QUARTER_TRIPLET,
    5: SIXTEENTH_DOTTED,
    6: EIGHTH,
    8: HALF_TRIPLET,
    9: EIGHTH_DOTTED,
    12: QUARTER,
    16: WHOLE_TRIPLET,
    18: QUARTER_DOTTED,
    24: HALF,
    36: HALF_DOTTED,
    48: WHOLE,
    72: WHOLE_DOTTED,
    }


def parseLeadsheets(ldir,wholeNoteDuration=128.0,verbose=False):
    """
    Parse a directory of leadsheets into a list of chords and melodies.

    Inputs:
        tdir: Path to directory containing leadsheets, i.e. "/transcriptions".
    """
    asserterror_count = 0
    keyerror_count = 0
    bigrest_count = 0
    numParsed = 0
    clist =[]
    mlist =[]
    poslist = []
    ckeylist = []
    namelist = []
    dlist = []
    if not ldir.endswith("/"):
        ldir += "/"
    if PITCH_REP == PITCH_MIDI:
        print("PITCH: MIDI")
    elif PITCH_REP == PITCH_INTERVAL:
        print("PITCH: INTERVAL")
    elif PITCH_REP == PITCH_CHORD:
        print("PITCH: CHORD")

    for filename in os.listdir(ldir):
        fdir = ldir+filename
        try:
            c,m=leadsheet.parse_leadsheet(fdir)
            mseq = []
            pseq = []
            cseq = []
            dseq = []
            index_count = 0
            valid_leadsheet = True
            slot = 0
            if PITCH_REP == PITCH_MIDI:
                for note in m:
                    cseq.append(c[index_count][0]) # get chord key for the slot
                    index_count+=1

                    if note[0] != None:
                        assert note[0] >= MIDI_MIN 
                        assert note[0] <= MIDI_MAX
                    restVal = MIDI_MAX+1
                    isRest = note[0] == None
                    nval = restVal if isRest else note[0]
                    if isRest and note[1] > 24 or note[1] > 48:
                        valid_leadsheet = False
                        break
                    actNoteVal = nval - MIDI_MIN # scale down to start at 0
                    if bits_or_onehot == BITS:
                        dur = int(round(note[1]*wholeNoteDuration/48.0))
                    elif bits_or_onehot == ONE_HOT:
                        try:
                            if note[1] not in DURATION_MAPPING.keys():
                                print("KEY ERROR: " + str(note[1]) + ". File: " + filename)
                            dur = DURATION_MAPPING[note[1]]
                        except KeyError:
                            if verbose:
                                print("KEY ERROR: " + str(note[1]) + ". File: " + filename)
                    mseq.append(actNoteVal)
                    dseq.append(dur)

                    pval_low = 0.0 if isRest else float(actNoteVal)/float(MIDI_MAX-MIDI_MIN)
                    pval_high = 0.0 if isRest else 1-pval_low
                    pseq.append((pval_high,pval_low))

                    for _ in range(note[1]-1):
                        cseq.append(c[index_count][0])
                        index_count+=1
                    if index_count >= SEQ_LEN:
                        break

            elif PITCH_REP == PITCH_INTERVAL:
                isStart = True
                for note in m:
                    if isStart and note[0] != None:
                        prevVal = note[0]-1
                        isStart = False
                    cseq.append(c[index_count][0]) # get chord key for the slot
                    index_count+=1

                    if note[0] != None:
                        assert note[0] >= MIDI_MIN 
                        assert note[0] <= MIDI_MAX
                    restVal = MAX_INTERV+1
                    isRest = note[0] == None
                    nval = restVal if isRest else note[0]-prevVal
                    nval = nval - MIN_INTERV
                    prevVal = prevVal if isRest else note[0]
                    if isRest and note[1] > 24 or note[1] > 48:
                        valid_leadsheet = False
                        break
                    if bits_or_onehot == BITS:
                        dur = int(round(note[1]*wholeNoteDuration/48.0))
                    elif bits_or_onehot == ONE_HOT:
                        dur = DURATION_MAPPING[note[1]]
                    mseq.append((nval,dur))

                    midival = MIDI_MAX+1 if isRest else note[0]
                    pval_low = 0.0 if isRest else float(midival-MIDI_MIN)/float(MIDI_MAX-MIDI_MIN)
                    pval_high = 0.0 if isRest else 1-pval_low
                    pseq.append((pval_high,pval_low))

                    for _ in range(note[1]-1):
                        cseq.append(c[index_count][0])
                        index_count+=1
                    if index_count >= SEQ_LEN:
                        break

            elif PITCH_REP == PITCH_CHORD:
                for note in m:
                    ckey = c[index_count][0]
                    cseq.append(ckey) # get chord key for the slot
                    index_count+=1

                    if note[0] != None:
                        assert note[0] >= MIDI_MIN 
                        assert note[0] <= MIDI_MAX
                    restVal = 12 # pitches go 0-11
                    isRest = note[0] == None
                    nval = restVal if isRest else (note[0]-ckey) % 12
                    if isRest and note[1] > 24 or note[1] > 48:
                        valid_leadsheet = False
                        break
                    if bits_or_onehot == BITS:
                        dur = int(round(note[1]*wholeNoteDuration/48.0))
                    elif bits_or_onehot == ONE_HOT:
                        dur = DURATION_MAPPING[note[1]]
                    mseq.append((nval,0))

                    midival = MIDI_MAX+1 if isRest else note[0]
                    pval_low = 0.0 if isRest else float(midival-MIDI_MIN)/float(MIDI_MAX-MIDI_MIN)
                    pval_high = 0.0 if isRest else 1-pval_low
                    pseq.append((pval_high,pval_low))

                    for _ in range(note[1]-1):
                        cseq.append(c[index_count][0])
                        index_count+=1
                    if index_count >= SEQ_LEN:
                        break

            if not valid_leadsheet:
                bigrest_count += 1
                continue
            numParsed += 1
            clist.append(c)
            mlist.append(mseq)
            poslist.append(pseq)
            ckeylist.append(cseq)
            namelist.append(filename)
            dlist.append(dseq)
        except KeyError:
            if verbose:
                print("KEY ERROR: "+filename)
            keyerror_count+=1
        except AssertionError:
            if verbose:
                print("ASSERT ERROR: " +filename)
            asserterror_count+=1
    if verbose:
        print("Num key errors: " + str(keyerror_count))
        print("Num assert errors: " + str(asserterror_count))
        print("Num leadsheets with too big rests: " + str(bigrest_count))
        print("Num leadsheets successfully parsed: " + str(numParsed))
    return [clist, mlist,poslist,ckeylist,namelist,dlist]

def saveLeadsheets(parsedLists,outputDirs):
    """
    Save parsed leadsheets.
    """
    for i in range(len(parsedLists)):
        parsedList = parsedLists[i]
        outputDir = outputDirs[i]
        with open(outputDir,'w') as outfile:
            json.dump(parsedList,outfile)

if __name__ == '__main__':
    parsedLists = parseLeadsheets(ldir,verbose=True)
    saveLeadsheets(parsedLists=parsedLists,outputDirs=outputDirList)