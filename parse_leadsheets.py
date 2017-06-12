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
susdir = parsedir + 'sus.json'
posdir = parsedir + 'pos.json'
ckeydir = parsedir + "chordkeys.json"
namedir = parsedir + 'names.json'

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
    suslist = []
    ckeylist = []
    namelist = []
    if not ldir.endswith("/"):
        ldir += "/"
    for filename in os.listdir(ldir):
        fdir = ldir+filename
        try:
            c,m=leadsheet.parse_leadsheet(fdir)
            mseq = []
            susseq = []
            cseq = []
            index_count = 0
            for note in m:
                cseq.append(c[index_count][0]) # get chord key for the slot
                if note[0] != None:
                    assert note[0] >= MIDI_MIN
                nval = note[0] if note[0] != None else MIDI_MAX+1
                mseq.append(nval-MIDI_MIN)
                susseq.append(0) # attack new note
                for _ in range(note[1]-1):
                    assert nval - MIDI_MIN >= 0
                    assert nval < MIDI_MAX+2
                    mseq.append(nval-MIDI_MIN)
                    susseq.append(1) # sustain for the remaining duration
                index_count+=1
            clist.append(c)
            mlist.append(mseq)
            suslist.append(susseq)
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
    return clist, mlist,suslist,ckeylist,namelist

def saveLeadsheets(clist,mlist,cdir,mdir):
    """
    Save parsed leadsheets.
    """
    with open(cdir,'w') as coutfile:
        json.dump(clist,coutfile)
    with open(mdir,'w') as moutfile:
        json.dump(mlist,moutfile)

if __name__ == '__main__':
    clist,mlist,suslist,ckeylist,namelist = parseLeadsheets(ldir,verbose=True)
    saveLeadsheets(clist,mlist,cdir,mdir)