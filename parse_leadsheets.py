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

def parseLeadsheets(ldir,verbose=False):
    """
    Parse a directory of leadsheets into a list of chords and melodies.

    Inputs:
        tdir: Path to directory containing leadsheets, i.e. "/transcriptions".
    """
    asserterror_count = 0
    keyerror_count = 0
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
            for note in m:
                cseq.append(c[index_count][0]) # get chord key for the slot
                if note[0] != None:
                    assert note[0] >= MIDI_MIN 
                restVal = MIDI_MAX+1
                susVal = MIDI_MAX+2-MIDI_MIN
                isRest = note[0] == None
                nval = restVal if isRest else note[0]
                actNoteVal = nval - MIDI_MIN # scale down to start at 0
                pval_low = 0.0 if isRest else float(actNoteVal)/float(MIDI_MAX)
                pval_high = 0.0 if isRest else 1-pval_low
                pseq.append((pval_high,pval_low))
                mseq.append(actNoteVal) # attack new note
                for _ in range(note[1]-1):
                    mseq.append(susVal) # sustain for rest of duration
                index_count+=1
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