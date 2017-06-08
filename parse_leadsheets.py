import leadsheet
import os
import json

ldir = "./ii-V-I_leadsheets/"
parsedir = "./parsed_ii-V-I_leadsheets/"
cdir = parsedir + "chords.json"
mdir = parsedir + 'melodies.json'

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
    namelist = []
    if not ldir.endswith("/"):
        ldir += "/"
    for filename in os.listdir(ldir):
        fdir = ldir+filename
        try:
            c,m=leadsheet.parse_leadsheet(fdir)
            mseq = []
            for note in m:
                nval = note[0] if note[0] != None else 128
                for _ in range(note[1]):
                    mseq.append(nval)
            clist.append(c)
            mlist.append(mseq)
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
    return clist, mlist

def saveLeadsheets(clist,mlist,cdir,mdir):
    """
    Save parsed leadsheets.
    """
    with open(cdir,'w') as coutfile:
        json.dump(clist,coutfile)
    with open(mdir,'w') as moutfile:
        json.dump(mlist,moutfile)

if __name__ == '__main__':
    clist,mlist = parseLeadsheets(ldir,verbose=True)
    saveLeadsheets(clist,mlist,cdir,mdir)