import leadsheet
import os
import json
import roadmap_parser
import constants
import copy


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
iiVIparsename = "ii-V-I_leadsheets"
FOUR_BAR_CUT = False
if USING_TRANSCRIPTIONS:
    parsename = "leadsheets_bricked_all2"
ldir = "./"+parsename+"/"
iiVIldir = "./"+iiVIparsename+"/"
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
alldir = parsedir + 'features.json'
outputDirList = [cdir,mdir,posdir,ckeydir,namedir,ddir,spdir]

MAXLENGTH = 20
MINLENGTH = 8

# Middle C is C5 = 60

MIDI_MIN = 48 # lowest note value found in trainingset
MIDI_MAX = 96 # highest note value found in trainingset
NUM_NOTES = MIDI_MAX-MIDI_MIN+1 # number of distinct notes in trainingset
if USING_TRANSCRIPTIONS:
    MIDI_MIN = 36#55#53#53#46#44#55
    MIDI_MAX = 108#82#89#89#96#106#84
    NUM_NOTES = MIDI_MAX-MIDI_MIN+1

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
    alist = []
    mlist_noskip = []
    poslist_noskip = []
    tlist_skip = []
    tlist_noskip = []
    if not ldir.endswith("/"):
        ldir += "/"
    if PITCH_REP == PITCH_MIDI:
        print("PITCH: MIDI")
    elif PITCH_REP == PITCH_INTERVAL:
        print("PITCH: INTERVAL")
    elif PITCH_REP == PITCH_CHORD:
        print("PITCH: CHORD")

    ##################### Get roadmaps to get chords & bricks

    # Clear temp directories
    folder = './temp_leadsheets/'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    # Generate roadmaps in temp directories
    max_length = 0 # For tracking purposes, usually 24 is good
    lowest_note = 200 # For tracking purposes
    highest_note = 0 # For tracking purposes
    roadmaps = roadmap_parser.generateRoadmaps(ldir,"./temp_leadsheets",doTranspose=False)

    ##################### Grab chords and durations from roadmaps
    # i (leadsheet index)
    # 0 ident (leadsheet), 1 roadmap for leadsheet
    # 0 ident (roadmap), 1 actual roadmap content
    # 0 blocks list
    # 0 ident (blocks), 1 actual blocks content
    # Chord durations are 480 for 1 measure
    rcdlist = []
    totalbrickdur = 0
    numChords = 0
    numBricks = 0
    possiblepitches = {}
    for j in range(len(roadmaps)):
        bricks = roadmaps[j][1][1][0][1]
        chordsdurs = []
        for brick in bricks:
            chordsdurs_brick = []
            chords = roadmap_parser.chordFinder(brick,durOn=True)
            numChords += len(chords)
            for i in range(len(chords)):
                #try:
                cval = leadsheet.parse_chord(chords[i][0])
                # except KeyError:
                #     print("KeyError at Roadmap ",j)
                #     print(os.listdir(ldir)[j])
                #     return
                dur = int(int(chords[i][1])/constants.RESOLUTION_SCALAR)
                totalbrickdur += dur
                chordsdurs_brick.append([cval,dur])
            chordsdurs.append(chordsdurs_brick)
        rcdlist.append(chordsdurs)
    
    #################### Parse features for each leadsheet
    roadmap_counter = -1
    for filename in os.listdir(ldir):
        """
        Parsing strategy:

        Want these features: clist, mlist,poslist,ckeylist,namelist,dlist,splist
        1. clist: chords, 1 per note
        2. mlist: note pitches
        3. poslist: list of (low,high) relative pitch positions
        4. ckeylist: list of pitches in the chord
        5. namelist: list of filenames
        6. dlist: list of durations (or beat positions)
        7. splist: list of starting note tuples (pitch, duration, beat pos, chord key, other dur)
        8. alist: list of attacks

        Parsing strategy:
        1. Parse leadsheet (segmented into 48 timeslots) into c (chord keys and notes), m (notes).
        2. Calculate pos for each m
        3. 
        """
        roadmap_counter += 1
        chordsdurs = rcdlist[roadmap_counter]
        fdir = ldir+filename
        try:

            # Parse leadsheet    
            c,m=leadsheet.parse_leadsheet(fdir)

        except KeyError:
            if verbose:
                print("KEY ERROR: "+filename)
            keyerror_count+=1
            continue
        except AssertionError:
            if verbose:
                print("ASSERT ERROR: " +filename)
            asserterror_count+=1
            continue
        try:
            totalnotedur = 0
            for note in m:
                totalnotedur+=note[1]
            brick_durcount = 0
            actual_chordsdurs = []
            brick_index = 0
            while brick_durcount <= totalnotedur:
                chordsdurs_brick = chordsdurs[brick_index % len(chordsdurs)]
                brick_index += 1
                actual_chordsdurs_brick = []
                for j in range(len(chordsdurs_brick)):
                    chorddur = chordsdurs_brick[j]
                    brick_dur = int(chorddur[1])
                    brick_c = chorddur[0]
                    if brick_c != c[brick_durcount % len(c)]:
                        continue
                    brick_durcount += brick_dur
                    actual_chordsdurs_brick.append(chorddur)
                actual_chordsdurs.append(actual_chordsdurs_brick)

            # Segment sequences by brick endtimes
            brick_endtimes = []
            timecount = 0
            for chordsdurs_brick in actual_chordsdurs:
                brick_dur = 0
                for chorddur in chordsdurs_brick:
                    brick_dur += int(chorddur[1])
                    pass
                timecount += int(brick_dur)
                brick_endtimes.append(timecount)
                pass
            print("Num bricks: ", len(brick_endtimes))
            print("Last endtime: ",brick_endtimes[-1])
            brick_count = 0 # Tracks brick count across leadsheet
            
            # Repeat chords until they match the notes for each timestep
            beat_positions = []
            attacks = []
            pitches = []
            pitches_skip = []
            clen = len(c)
            totaldur = 0
            seq_noskip = []
            seq_skip = []
            seqs_noskip = []
            seqs_skip = []
            for note in m:
                #print(str(brick_count) + "     " + str(len(seqs_skip)))
                pitch = note[0] if note[0] != None else MIDI_MAX+1
                pitch -= MIDI_MIN
                dur = note[1]
                totaldur += dur
                # Save general stats
                beat_positions.append(totaldur % int(constants.WHOLE/constants.RESOLUTION_SCALAR))
                attacks.append(1)
                pitches_skip.append(pitch)
                pitches.append(pitch)

                # Save segmented brick info
                endtime = brick_endtimes[brick_count]
                if totaldur > endtime:
                    # We need to break this note into subparts to fit the bricks
                    # End sequence at brick endtime
                    bp = endtime % int(constants.WHOLE/constants.RESOLUTION_SCALAR)
                    if bp in [11,17,23,29]:
                        print("WUT1",bp)
                    seq_skip.append([pitch, endtime % int(constants.WHOLE/constants.RESOLUTION_SCALAR),totaldur-dur])
                    seq_noskip.append([pitch,1,totaldur-dur])
                    for k in range(endtime-(totaldur-dur)-1):
                        seq_noskip.append([pitch,0,totaldur-dur+k+1])
                    # Save sequences and resetm 
                    seqs_skip.append(seq_skip)
                    seqs_noskip.append(seq_noskip)
                    if max_length < len(seq_skip):# and len(seq_skip) < 30:
                        max_length = len(seq_skip)
                    seq_skip = []
                    seq_noskip = []
                    brick_count += 1
                    # Add leftovers
                    bp = (totaldur-endtime) % int(constants.WHOLE/constants.RESOLUTION_SCALAR)
                    if bp in [11,17,23,29]:
                        print("WUT2",bp)
                    seq_skip.append([pitch, (totaldur-endtime) % int(constants.WHOLE/constants.RESOLUTION_SCALAR), endtime])
                    seq_noskip.append([pitch,1,endtime])
                    for k in range(totaldur-endtime-1):
                        seq_noskip.append([pitch,0,endtime+k+1])                    
                    # Update general attacks/pitches
                    for _ in range(endtime-(totaldur-dur)-1):
                        attacks.append(0)
                        pitches.append(pitch)
                    attacks.append(1)
                    pitches.append(pitch)
                    for _ in range(totaldur - endtime - 1):
                        attacks.append(0)
                        pitches.append(pitch)                        

                else:
                    # Update sequences as normal
                    bp = totaldur % int(constants.WHOLE/constants.RESOLUTION_SCALAR)
                    if bp in [11,17,23,29]:
                        print("WUT3",bp)
                    seq_skip.append([pitch, totaldur % int(constants.WHOLE/constants.RESOLUTION_SCALAR),totaldur-dur])
                    seq_noskip.append([pitch,1,totaldur-dur])
                    for k in range(dur-1):
                        seq_noskip.append([pitch, 0,totaldur-dur+k+1])
                    # Update general attacks/pitches
                    for _ in range(dur-1):
                        attacks.append(0)
                        pitches.append(pitch)

                if totaldur == endtime:
                    # Brick finished: Save current sequences and reset
                    seqs_skip.append(seq_skip)
                    seqs_noskip.append(seq_noskip)
                    if max_length < len(seq_skip):# and len(seq_skip) < 30:
                        max_length = len(seq_skip)

                    seqdur = 0
                    if len(seq_skip)>4:
                        prevbeatpos = seq_skip[4][1]
                        for pitch,beatpos,index in seq_skip[4:]:
                            seqdur += (beatpos - prevbeatpos+48) % 48
                            prevbeatpos=beatpos
                    if len(seq_skip) > 4 and seqdur < 48*4:
                        for pitch,beatpos,index in seq_skip:
                            note = [pitch]
                            if note[0] != MIDI_MAX-MIDI_MIN+1 and note[0] < lowest_note:
                                lowest_note = note[0]
                            if note[0]!=MIDI_MAX-MIDI_MIN+1 and note[0] > highest_note:
                                highest_note = note[0]

                    seq_skip = []
                    seq_noskip = []
                    brick_count += 1


            for i in range(totaldur-clen):
                c.append(c[i % clen])
            print("Last index: ",len(c))
            sps = []
            for i in range(4):
                note = m[i]
                noteval = MIDI_MAX+1-MIDI_MIN if note[0] == None else note[0]-MIDI_MIN
                dur = note[1]
                spi = [noteval, beat_positions[i],dur,c[i][0],dur]
                if beat_positions[i] in [11,17,23,29]:
                    print("WUT", beat_positions[i])
                sps.append(spi)

            # Transpose everything
            newc = []
            newpitches_skip = []
            newp_skip = []
            newckeys = []
            newfilenames =[]
            newbeats = []
            newsps = []
            newattacks = []
            newpitches = []
            newp = []
            transposed_seqs_skip = []
            transposed_seqs_noskip = []
            for keydiff in range(12):     
                holder = []
                for i in range(len(seqs_skip)):
                    seq_skip = copy.deepcopy(seqs_skip[i])
                    for j in range(len(seq_skip)):
                        note = seq_skip[j][0]
                        if note != MIDI_MAX-MIDI_MIN+1:
                            seq_skip[j][0] = (note+keydiff if (note+keydiff <= MIDI_MAX- MIDI_MIN) else note+keydiff-12)
                        #if keydiff == 1:
                        if keydiff < 6:
                            if note not in possiblepitches.keys():
                                possiblepitches[note]=0
                            possiblepitches[note] += 1
                    holder.append(seq_skip)
                transposed_seqs_skip.append(holder)
                holder = []
                for i in range(len(seqs_noskip)):
                    seq_noskip = copy.deepcopy(seqs_noskip[i])
                    for j in range(len(seq_noskip)):
                        note = seq_noskip[j][0]
                        if note != MIDI_MAX-MIDI_MIN+1:
                            seq_noskip[j][0] = (note+keydiff if (note+keydiff <= MIDI_MAX- MIDI_MIN) else note+keydiff-12)
                    holder.append(seq_noskip)
                transposed_seqs_noskip.append(holder)
                newctemp = [((x[0]+keydiff) % 12,x[1]) for x in c]
                newc.append(newctemp)
                newckeys.append([ctuple[0] for ctuple in newctemp])
                newfilenames.append(filename)
                newbeats.append(beat_positions)
                newattacks.append(attacks)
                newpitches_skiptemp = [((note+keydiff if (note+keydiff <= MIDI_MAX- MIDI_MIN) else note+keydiff-12) if note != MIDI_MAX-MIDI_MIN+1 else note) for note in pitches_skip]
                newpitchestemp = [((note+keydiff if (note+keydiff <= MIDI_MAX- MIDI_MIN) else note+keydiff-12) if note != MIDI_MAX-MIDI_MIN+1 else note) for note in pitches]
                newpitches_skip.append(newpitches_skiptemp)
                newpitches.append(newpitchestemp)
                newp_skip.append([((0.0,0.0) if (note == MIDI_MAX-MIDI_MIN+1) else (float(note/float(MIDI_MAX-MIDI_MIN)),1.0-float(note/float(MIDI_MAX-MIDI_MIN))) ) for note in newpitches_skiptemp])
                newp.append([((0.0,0.0) if (note == MIDI_MAX-MIDI_MIN+1) else (float(note/float(MIDI_MAX-MIDI_MIN)),1.0-float(note/float(MIDI_MAX-MIDI_MIN))) ) for note in newpitchestemp])
                
                newspstemp = []
                for i in range(4):
                    newsp = sps[i]
                    if newsp[0] != MIDI_MAX-MIDI_MIN+1:
                        newsp[0] = sps[i][0]+keydiff if (sps[i][0]+keydiff <= MIDI_MAX-MIDI_MIN) else sps[i][0]+keydiff-12
                    newsp[3] = (sps[i][3]+keydiff) % 12
                    newspstemp.append(newsp)               
                newsps.append(newspstemp)
            clist.append(newc)
            mlist.append(newpitches_skip)
            poslist.append(newp_skip)
            ckeylist.append([ctuple[0] for ctuple in newc])
            namelist.append(filename)
            for bp in beat_positions:
                if bp in [11,17,23,29]:
                    print("WUT",bp)
            dlist.append(beat_positions)
            splist.append(newsps)
            alist.append(attacks)
            mlist_noskip.append(newpitches)
            poslist_noskip.append(newp)
            tlist_skip.append(transposed_seqs_skip)
            tlist_noskip.append(transposed_seqs_noskip)
            numParsed += len(seqs_skip)
        except KeyError:
            print(filename)
    if verbose:
        print("Num key errors: " + str(keyerror_count))
        print("Num assert errors: " + str(asserterror_count))
        #print("Num bad base phrases (short, long, error): " + str(bigrest_count))
        print("Num phrases successfully parsed: " + str(numParsed))
        print("Max length: " + str(max_length))
        print("Highest note: " + str(highest_note))
        print("Lowest note: " + str(lowest_note))
        keys = list(possiblepitches.keys())
        keys.sort()
        for key in keys:
            print(key,possiblepitches[key])


    # Thus far, we have parsed the leadsheets from transcriptions. We now parse the leadsheets from ii-V-Is.
    #TODO
    asserterror_count = 0
    keyerror_count = 0
    bigrest_count = 0
    numParsed = 0
    max_length = 0 # For tracking purposes, usually 24 is good
    lowest_note = 200 # For tracking purposes
    highest_note = 0 # For tracking purposes
    possiblepitches = {}
    for filename in os.listdir(iiVIldir):
        fdir = iiVIldir+filename
        try:
            # Parse leadsheet    
            c,m=leadsheet.parse_leadsheet(fdir)

        except KeyError:
            if verbose:
                print("KEY ERROR: "+filename)
            keyerror_count+=1
            continue
        except AssertionError:
            if verbose:
                print("ASSERT ERROR: " +filename)
            asserterror_count+=1
            continue
        try:            
            # Repeat chords until they match the notes for each timestep
            beat_positions = []
            attacks = []
            pitches = []
            pitches_skip = []
            clen = len(c)
            totaldur = 0
            seq_noskip = []
            seq_skip = []
            seqs_noskip = []
            seqs_skip = []
            for note in m:
                pitch = note[0] if note[0] != None else MIDI_MAX+1
                pitch -= MIDI_MIN
                dur = note[1]
                totaldur += dur
                # Save general stats
                beat_positions.append(totaldur % int(constants.WHOLE/constants.RESOLUTION_SCALAR))
                attacks.append(1)
                pitches_skip.append(pitch)
                pitches.append(pitch)        

                bp = totaldur % int(constants.WHOLE/constants.RESOLUTION_SCALAR)
                if bp in [11,17,23,29]:
                    print("WUT3",bp)
                seq_skip.append([pitch, totaldur % int(constants.WHOLE/constants.RESOLUTION_SCALAR),totaldur-dur])
                seq_noskip.append([pitch,1,totaldur-dur])
                for k in range(dur-1):
                    seq_noskip.append([pitch, 0,totaldur-dur+k+1])
                # Update general attacks/pitches
                for _ in range(dur-1):
                    attacks.append(0)
                    pitches.append(pitch)

            seqs_skip.append(seq_skip)
            seqs_noskip.append(seq_noskip)
            if max_length < len(seq_skip):# and len(seq_skip) < 30:
                max_length = len(seq_skip)

            seqdur = 0
            if len(seq_skip)>4:
                prevbeatpos = seq_skip[4][1]
                for pitch,beatpos,index in seq_skip[4:]:
                    seqdur += (beatpos - prevbeatpos+48) % 48
                    prevbeatpos=beatpos
            if len(seq_skip) > 4 and seqdur < 48*4:
                for pitch,beatpos,index in seq_skip:
                    note = [pitch]
                    if note[0] != MIDI_MAX-MIDI_MIN+1 and note[0] < lowest_note:
                        lowest_note = note[0]
                    if note[0]!=MIDI_MAX-MIDI_MIN+1 and note[0] > highest_note:
                        highest_note = note[0]

            for i in range(totaldur-clen):
                c.append(c[i % clen])
            print("Last index: ",len(c))
            sps = []
            for i in range(4):
                note = m[i]
                noteval = MIDI_MAX+1-MIDI_MIN if note[0] == None else note[0]-MIDI_MIN
                dur = note[1]
                spi = [noteval, beat_positions[i],dur,c[i][0],dur]
                if beat_positions[i] in [11,17,23,29]:
                    print("WUT", beat_positions[i])
                sps.append(spi)

            # Transpose everything
            newc = []
            newpitches_skip = []
            newp_skip = []
            newckeys = []
            newfilenames =[]
            newbeats = []
            newsps = []
            newattacks = []
            newpitches = []
            newp = []
            transposed_seqs_skip = []
            transposed_seqs_noskip = []
            for keydiff in range(12):     
                holder = []
                for i in range(len(seqs_skip)):
                    seq_skip = copy.deepcopy(seqs_skip[i])
                    for j in range(len(seq_skip)):
                        note = seq_skip[j][0]
                        if note != MIDI_MAX-MIDI_MIN+1:
                            seq_skip[j][0] = (note+keydiff if (note+keydiff <= MIDI_MAX- MIDI_MIN) else note+keydiff-12)
                        #if keydiff == 1:
                        if keydiff < 6:
                            if note not in possiblepitches.keys():
                                possiblepitches[note]=0
                            possiblepitches[note] += 1
                    holder.append(seq_skip)
                transposed_seqs_skip.append(holder)
                holder = []
                for i in range(len(seqs_noskip)):
                    seq_noskip = copy.deepcopy(seqs_noskip[i])
                    for j in range(len(seq_noskip)):
                        note = seq_noskip[j][0]
                        if note != MIDI_MAX-MIDI_MIN+1:
                            seq_noskip[j][0] = (note+keydiff if (note+keydiff <= MIDI_MAX- MIDI_MIN) else note+keydiff-12)
                    holder.append(seq_noskip)
                transposed_seqs_noskip.append(holder)
                newctemp = [((x[0]+keydiff) % 12,x[1]) for x in c]
                newc.append(newctemp)
                newckeys.append([ctuple[0] for ctuple in newctemp])
                newfilenames.append(filename)
                newbeats.append(beat_positions)
                newattacks.append(attacks)
                newpitches_skiptemp = [((note+keydiff if (note+keydiff <= MIDI_MAX- MIDI_MIN) else note+keydiff-12) if note != MIDI_MAX-MIDI_MIN+1 else note) for note in pitches_skip]
                newpitchestemp = [((note+keydiff if (note+keydiff <= MIDI_MAX- MIDI_MIN) else note+keydiff-12) if note != MIDI_MAX-MIDI_MIN+1 else note) for note in pitches]
                newpitches_skip.append(newpitches_skiptemp)
                newpitches.append(newpitchestemp)
                newp_skip.append([((0.0,0.0) if (note == MIDI_MAX-MIDI_MIN+1) else (float(note/float(MIDI_MAX-MIDI_MIN)),1.0-float(note/float(MIDI_MAX-MIDI_MIN))) ) for note in newpitches_skiptemp])
                newp.append([((0.0,0.0) if (note == MIDI_MAX-MIDI_MIN+1) else (float(note/float(MIDI_MAX-MIDI_MIN)),1.0-float(note/float(MIDI_MAX-MIDI_MIN))) ) for note in newpitchestemp])
                
                newspstemp = []
                for i in range(4):
                    newsp = sps[i]
                    if newsp[0] != MIDI_MAX-MIDI_MIN+1:
                        newsp[0] = sps[i][0]+keydiff if (sps[i][0]+keydiff <= MIDI_MAX-MIDI_MIN) else sps[i][0]+keydiff-12
                    newsp[3] = (sps[i][3]+keydiff) % 12
                    newspstemp.append(newsp)               
                newsps.append(newspstemp)
            clist.append(newc)
            mlist.append(newpitches_skip)
            poslist.append(newp_skip)
            ckeylist.append([ctuple[0] for ctuple in newc])
            namelist.append(filename)
            for bp in beat_positions:
                if bp in [11,17,23,29]:
                    print("WUT",bp)
            dlist.append(beat_positions)
            splist.append(newsps)
            alist.append(attacks)
            mlist_noskip.append(newpitches)
            poslist_noskip.append(newp)
            tlist_skip.append(transposed_seqs_skip)
            tlist_noskip.append(transposed_seqs_noskip)
            numParsed += len(seqs_skip)
        except KeyError:
            print(filename)

    if verbose:
        print("Num key errors: " + str(keyerror_count))
        print("Num assert errors: " + str(asserterror_count))
        #print("Num bad base phrases (short, long, error): " + str(bigrest_count))
        print("Num phrases successfully parsed: " + str(numParsed))
        print("Max length: " + str(max_length))
        print("Highest note: " + str(highest_note))
        print("Lowest note: " + str(lowest_note))
        keys = list(possiblepitches.keys())
        keys.sort()
        for key in keys:
            print(key,possiblepitches[key])


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
            "transposed_seqs_skip":tlist_skip,
            "transposed_seqs_noskip":tlist_noskip}

def saveLeadsheets(features):
    """
    Save parsed leadsheets.
    """
    if not os.path.exists(parsedir):
        print("CREATED WRITE DIR")
        os.makedirs(parsedir)
    if not os.path.isdir(parsedir):
        print("ERROR: WRITE DIR DOES NOT EXIST")
    with open(alldir,'w') as outfile:
        json.dump(features,outfile)

if __name__ == '__main__':
    features = parseLeadsheets(ldir,verbose=True)
    saveLeadsheets(features)
