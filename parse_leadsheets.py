import leadsheet
import os
import json

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
parsedir = "./parsed_ii-V-I_leadsheets/" + category
cdir = parsedir + "chords.json"
mdir = parsedir + 'melodies.json'
posdir = parsedir + 'pos.json'
ckeydir = parsedir + "chordkeys.json"
namedir = parsedir + 'names.json'
outputDirList = [cdir,mdir,posdir,ckeydir,namedir]

MIDI_MIN = 55 # lowest note value found in trainingset
MIDI_MAX = 89 # highest note value found in trainingset
NUM_NOTES = MIDI_MAX-MIDI_MIN+1 # number of distinct notes in trainingset
SEQ_LEN = 96

MIN_INTERV = -13
MAX_INTERV = 14


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
            index_count = 0
            valid_leadsheet = True
            slot = 0
            if PITCH_REP == PITCH_MIDI:
                for note in m:
                    cseq.append(c[index_count][0]) # get chord key for the slot
                    index_count+=1
                    if note[0] != None:
                        assert note[0] >= MIDI_MIN 
                    restVal = MIDI_MAX+1
                    isRest = note[0] == None
                    nval = restVal if isRest else note[0]
                    if isRest and note[1] > 12 or note[1] > 24:
                        valid_leadsheet = False
                        break
                    actNoteVal = nval - MIDI_MIN # scale down to start at 0
                    pval_low = 0.0 if isRest else float(actNoteVal)/float(MIDI_MAX-MIDI_MIN)
                    pval_high = 0.0 if isRest else 1-pval_low
                    pseq.append((pval_high,pval_low))
                    mseq.append((actNoteVal,0)) # attack new note
                    for _ in range(note[1]-1):
                        pseq.append((pval_high,pval_low))
                        mseq.append((actNoteVal,1)) # sustain for rest of duration
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
                    restVal = MAX_INTERV+1
                    isRest = note[0] == None
                    nval = restVal if isRest else note[0]-prevVal
                    nval = nval - MIN_INTERV
                    prevVal = prevVal if isRest else note[0]
                    if isRest and note[1] > 12 or note[1] > 24:
                        valid_leadsheet = False
                        break
                    midival = MIDI_MAX+1 if isRest else note[0]
                    pval_low = 0.0 if isRest else float(midival-MIDI_MIN)/float(MIDI_MAX-MIDI_MIN)
                    pval_high = 0.0 if isRest else 1-pval_low
                    pseq.append((pval_high,pval_low))
                    mseq.append((nval,0))
                    for _ in range(note[1]-1):
                        pseq.append((pval_high,pval_low))
                        mseq.append((0- MIN_INTERV,1)) #sustain
                        cseq.append(c[index_count][0])
                        index_count+=1
                    if index_count >= SEQ_LEN:
                        break
            elif PITCH_REP == PITCH_CHORD:
                for note in m:
                    ckey = c[index_count][0]
                    cseq.append(ckey) # get chord key for the slot
                    index_count+=1
                    restVal = 12 # pitches go 0-11
                    isRest = note[0] == None
                    nval = restVal if isRest else (note[0]-ckey) % 12
                    if isRest and note[1] > 12 or note[1] > 24:
                        valid_leadsheet = False
                        break
                    midival = MIDI_MAX+1 if isRest else note[0]
                    pval_low = 0.0 if isRest else float(midival-MIDI_MIN)/float(MIDI_MAX-MIDI_MIN)
                    pval_high = 0.0 if isRest else 1-pval_low
                    pseq.append((pval_high,pval_low))
                    mseq.append((nval,0))
                    for _ in range(note[1]-1):
                        pseq.append((pval_high,pval_low))
                        mseq.append((nval,1))
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
    return [clist, mlist,poslist,ckeylist,namelist]

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