import leadsheet
import os
import json

INTERVAL = True
CHORD = False
interval_or_chord = INTERVAL
ldir = "./ii-V-I_leadsheets/"
category = "intervalexpert_" if interval_or_chord else "chordexpert_"
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
            for note in m:
                cseq.append(c[index_count][0]) # get chord key for the slot
                if note[0] != None:
                    assert note[0] >= MIDI_MIN 
                restVal = MIDI_MAX+1
                isRest = note[0] == None
                nval = restVal if isRest else note[0]
                if isRest and note[1] > 12 or note[1] > 24:
                    valid_leadsheet = False
                    break
                actNoteVal = nval - MIDI_MIN # scale down to start at 0
                pval_low = 0.0 if isRest else float(actNoteVal)/float(MIDI_MAX)
                pval_high = 0.0 if isRest else 1-pval_low
                pseq.append((pval_high,pval_low))
                mseq.append((actNoteVal,0)) # attack new note
                for _ in range(note[1]-1):
                    mseq.append((actNoteVal,1)) # sustain for rest of duration
                index_count+=1
                slot += note[1]
                if slot >= SEQ_LEN:
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