import leadsheet
import os
import json


USING_TRANSCRIPTIONS = True
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
parsename = "ii-V-I_leadsheets"
if USING_TRANSCRIPTIONS:
    parsename = "transcriptions"
ldir = "./"+parsename+"/"
category = "pitchexpert_" if PITCH_REP == PITCH_MIDI else ("intervalexpert_" if PITCH_REP == PITCH_INTERVAL else "chordexpert_")
encoding = "bit_" if bits_or_onehot == BITS else "onehot_"
parsedir = "./parsed_"+parsename+"_dur/" + category + encoding
cdir = parsedir + "chords.json"
mdir = parsedir + 'melodies.json'
posdir = parsedir + 'pos.json'
ckeydir = parsedir + "chordkeys.json"
namedir = parsedir + 'names.json'
ddir = parsedir + 'durs.json'
spdir = parsedir + 'startpitches.json'
outputDirList = [cdir,mdir,posdir,ckeydir,namedir,ddir,spdir]

MIDI_MIN = 55 # lowest note value found in trainingset
MIDI_MAX = 89 # highest note value found in trainingset
NUM_NOTES = MIDI_MAX-MIDI_MIN+1 # number of distinct notes in trainingset
if USING_TRANSCRIPTIONS:
    MIDI_MIN = 46#44#55
    MIDI_MAX = 96#106#84
    NUM_NOTES = MIDI_MAX-MIDI_MIN+1

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

NUM_BITS = 7
WHOLE_NOTE_DURATION = 64.0
NOTES_PER_SEQ = 16
JUMP = 4

def parseLeadsheets(ldir,verbose=False):
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
    splist = []
    if not ldir.endswith("/"):
        ldir += "/"
    if PITCH_REP == PITCH_MIDI:
        print("PITCH: MIDI")
    elif PITCH_REP == PITCH_INTERVAL:
        print("PITCH: INTERVAL")
    elif PITCH_REP == PITCH_CHORD:
        print("PITCH: CHORD")

    max_length = 0
    notes_captured = {}
    lowest_note = 200
    highest_note = 0
    for filename in os.listdir(ldir):
        fdir = ldir+filename
        print(fdir)
        try:
            c,m=leadsheet.parse_leadsheet(fdir)
            index_count = 0
            skip_count = 0
            note_count = 0
            clen = len(c)
            print(len(c))
            print(len(m))
            totaldur = 0
            for n in m:
                totaldur+=n[1]
            print(totaldur)
            lenm = len(m)
            while note_count < lenm:
                mseq = []
                pseq = []
                cseq = []
                ckeyseq = []
                dseq = []
                valid_leadsheet = True
                if PITCH_REP == PITCH_MIDI:
                    isStart = True
                    while note_count < lenm:
                        note = m[note_count]
                        note_count += 1
                        dur = -1
                        if bits_or_onehot == BITS:
                            newdur = int(round(note[1]*WHOLE_NOTE_DURATION/48.0))
                            bindur = bin(newdur)[2:]
                            assert len(bindur) <= 7
                            bindur = '0'*(NUM_BITS-len(bindur))+bindur
                            dur = [int(x) for x in bindur]
                        elif bits_or_onehot == ONE_HOT:
                            try:
                                #if note[1] not in DURATION_MAPPING.keys():
                                #    print("KEY ERROR: " + str(note[1]) + ". File: " + filename)
                                assert note[1] <= 48
                                dur = note[1]-1#DURATION_MAPPING[note[1]]
                            except AssertionError:
                                keyerror_count += 1
                                valid_leadsheet = False
                                break
                                if verbose:
                                    print("KEY ERROR: " + str(note[1]) + ". File: " + filename)
                        if isStart and note[0] != None:
                            splist_tuple = (note[0]-MIDI_MIN,dur,index_count % 48,c[index_count % clen][0])
                            index_count += note[1]
                            isStart = False
                            continue
                        elif isStart and note[0] == None:
                            index_count += note[1]
                            skip_count += note[1]
                            continue

                        if note[0] == None and note[1] >= 12:
                            break # found rest, end of phrase

                        cseq.append(c[index_count % clen]) # get the full chord vec for the slot
                        ckeyseq.append(c[index_count % clen][0]) # get chord key for the slot

                        if note[0] != None:
                            if note[0] > highest_note:
                                highest_note = note[0]
                            if note[0] < lowest_note:
                                lowest_note = note[0]
                            assert note[0] >= MIDI_MIN
                            assert note[0] <= MIDI_MAX
                        restVal = MIDI_MAX+1
                        isRest = note[0] == None
                        nval = restVal if isRest else note[0]
                        #if note[1] > 48:
                        #    valid_leadsheet = False
                        #    break
                        actNoteVal = nval - MIDI_MIN # scale down to start at 0
                        mseq.append(actNoteVal)
                        dseq.append(dur)
                        index_count+=1

                        pval_low = 0.0 if isRest else float(actNoteVal)/float(MIDI_MAX-MIDI_MIN)
                        pval_high = 0.0 if isRest else 1-pval_low
                        pseq.append((pval_high,pval_low))

                        for _ in range(note[1]-1):
                            index_count+=1 # skip chords for sustain
                        #if index_count-skip_count >= SEQ_LEN:
                        #    break

                elif PITCH_REP == PITCH_INTERVAL:
                    prevVal = None
                    isStart = True
                    for note in m:
                        if bits_or_onehot == BITS:
                            dur = int(round(note[1]*WHOLE_NOTE_DURATION/48.0))
                        elif bits_or_onehot == ONE_HOT:
                            try:
                                if note[1] not in DURATION_MAPPING.keys():
                                    print("KEY ERROR: " + str(note[1]) + ". File: " + filename)
                                dur = DURATION_MAPPING[note[1]]
                            except KeyError:
                                keyerror_count += 1
                                valid_leadsheet = False
                                break
                                if verbose:
                                    print("KEY ERROR: " + str(note[1]) + ". File: " + filename)
                        if isStart and note[0] != None:
                            prevVal = note[0]
                            splist.append((note[0]-MIDI_MIN,dur))
                            index_count += note[1]
                            isStart = False
                            continue
                        elif isStart and note[0] == None:
                            index_count += note[1]
                            skip_count += note[1]
                            continue
                        cseq.append(c[index_count])
                        ckeyseq.append(c[index_count][0]) # get chord key for the slot

                        if note[0] != None:
                            assert note[0] >= MIDI_MIN 
                            assert note[0] <= MIDI_MAX
                        restVal = MAX_INTERV+1
                        isRest = note[0] == None
                        nval = restVal if isRest else note[0]-prevVal
                        nval = nval - MIN_INTERV # normalize to 0
                        prevVal = prevVal if isRest else note[0]
                        #if note[1] > 48:
                        #    valid_leadsheet = False
                        #    break
                        mseq.append(nval)
                        if nval in notes_captured.keys():
                            notes_captured[nval] += 1
                        else:
                            notes_captured[nval] = 1
                        dseq.append(dur)
                        index_count+=1

                        midival = MIDI_MAX+1 if isRest else note[0]
                        pval_low = 0.0 if isRest else float(midival-MIDI_MIN)/float(MIDI_MAX-MIDI_MIN)
                        pval_high = 0.0 if isRest else 1-pval_low
                        pseq.append((pval_high,pval_low))

                        for _ in range(note[1]-1):
                            index_count+=1
                        if index_count >= SEQ_LEN:
                            break

                elif PITCH_REP == PITCH_CHORD:
                    for note in m:
                        cseq = c[index_count]
                        ckey = c[index_count][0]
                        ckeyseq.append(ckey) # get chord key for the slot
                        index_count+=1

                        if note[0] != None:
                            assert note[0] >= MIDI_MIN 
                            assert note[0] <= MIDI_MAX
                        restVal = 12 # pitches go 0-11
                        isRest = note[0] == None
                        nval = restVal if isRest else (note[0]-ckey) % 12
                        #if isRest and note[1] > 24 or note[1] > 48:
                        #    valid_leadsheet = False
                        #    break
                        if bits_or_onehot == BITS:
                            dur = int(round(note[1]*WHOLE_NOTE_DURATION/48.0))
                        elif bits_or_onehot == ONE_HOT:
                            dur = DURATION_MAPPING[note[1]]
                        mseq.append(nval)
                        dseq.append(dur)

                        midival = MIDI_MAX+1 if isRest else note[0]
                        pval_low = 0.0 if isRest else float(midival-MIDI_MIN)/float(MIDI_MAX-MIDI_MIN)
                        pval_high = 0.0 if isRest else 1-pval_low
                        pseq.append((pval_high,pval_low))

                        for _ in range(note[1]-1):
                            index_count+=1
                        if index_count >= SEQ_LEN:
                            break

                if not valid_leadsheet or isStart or len(mseq) < 10 or len(mseq) > 30:
                    bigrest_count += 1
                    continue

                for keydiff in range(12):
                    newc = [((x[0]+keydiff) % 12,x[1]) for x in c]
                    newm = [(note+keydiff if (note+keydiff <= MIDI_MAX- MIDI_MIN) else note+keydiff-12) for note in mseq]
                    newp = [((0.0,0.0) if (note == MIDI_MAX-MIDI_MIN+1) else (float(note/float(MIDI_MAX-MIDI_MIN)),1.0-float(note/float(MIDI_MAX-MIDI_MIN))) ) for note in newm]
                    newsplist_tuple = list(splist_tuple)
                    newsplist_tuple[0] = newsplist_tuple[0]+keydiff if (newsplist_tuple[0]+keydiff <= MIDI_MAX - MIDI_MIN) else newsplist_tuple[0]+keydiff-12
                    newsplist_tuple[3] = (newsplist_tuple[3]+keydiff) % 12
                    numParsed += 1
                    clist.append(newc)
                    mlist.append(newm)
                    if max_length < len(mseq):
                        max_length = len(mseq)
                    poslist.append(newp)
                    ckeylist.append(ckeyseq)
                    namelist.append(filename)
                    dlist.append(dseq)
                    splist.append(newsplist_tuple)
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
        print("Max length: " + str(max_length))
        print("Highest note: " + str(highest_note))
        print("Lowest note: " + str(lowest_note))
        print(notes_captured)
    return [clist, mlist,poslist,ckeylist,namelist,dlist,splist]

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