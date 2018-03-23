
import json
import os
import leadsheet

valid_temps = [3,6,12,24,48,1,2,4,8,16,0]
valids = []
for r1 in valid_temps:
    for r2 in valid_temps:
        rcalc = r1+r2
        if rcalc not in valids:
            valids.append(rcalc)
def calcDurs_training(seqs_dur,s):
    act_seq_durs = []
    for i in range(len(seqs_dur)):
        act_durs = []
        seq_dur = seqs_dur[i]
        prev_beatpos = s[i][0][1]
        for beatpos in seq_dur:
            dur = (beatpos-prev_beatpos+48-1)%48+1
            act_durs.append(dur)
            prev_beatpos = beatpos
        act_seq_durs.append(act_durs)
    return act_seq_durs
    
def calcDurs(seqs_dur,n0s):
    act_seq_durs = []
    for i in range(len(seqs_dur)):
        act_durs = []
        seq_dur = seqs_dur[i]
        prev_beatpos = n0s[i][1]
        for beatpos in seq_dur:
            dur = (beatpos-prev_beatpos+48-1)%48+1
            act_durs.append(dur)
            prev_beatpos = beatpos
        act_seq_durs.append(act_durs)
    return act_seq_durs
def calcQR(act_seq_durs):
    validcount=0.0
    totalcount=0.0
    for seq in act_seq_durs:
        for dur in seq:
            if dur in valids:
                validcount+=1
            totalcount+=1
    return validcount/totalcount
def calcCPR(seqs,l):
    numConsec=0.0
    total = 0.0
    for seqi in range(len(seqs)):
        seq = seqs[seqi]
        repCount = 0
        index = 0
        seqpitch = -1
        while index <= len(seq):
            if index == len(seq):
                if repCount >= l:
                    numConsec+=1
                total+=1
                break
            if seqpitch == -1:
                seqpitch = seq[index]
                repCount = 1
            elif seqpitch != seq[index]:
                if repCount >= l:
                    numConsec+=1
                total+=1
                seqpitch = seq[index]
                repCount = 1
            else:
                repCount += 1
            index+=1
    return numConsec/total
def calcRPD(seqs,act_seq_durs,d):
    numConsec=0.0
    total = 0.0
    for seqi in range(len(seqs)):
        seq = seqs[seqi]
        act_durs = act_seq_durs[seqi]
        repCount = 0
        index = 0
        seqdur = 0
        seqpitch = -1
        while index <= len(seq):
            if index == len(seq):
                if repCount > 1:
                    if seqdur >= d:
                        numConsec+=1
                total+=1
                break
            if seqpitch == -1:
                seqpitch = seq[index]
                seqdur = act_durs[index]
                repCount = 1
            elif seqpitch != seq[index]:
                if repCount > 1:
                    if seqdur >= d:
                        numConsec+=1
                total+=1
                seqpitch = seq[index]
                seqdur = act_durs[index]
                repCount = 1
            else:
                seqdur += act_durs[index]
                repCount += 1
            index+=1
    return numConsec/total
def calcTS(seqs,d,REST_VAL):
    count =0.0
    total =0.0
    for seq in seqs:
        basePitch = seq[0]
        for i in range(len(seq)-1):
            i=i+1
            if abs(seq[i]-basePitch) > d and basePitch!=REST_VAL and seq[i]!=REST_VAL:
                count += 1
            total+=1
            basePitch = seq[i]
    return count/total
def calcOR(seqs_dur,l):
    validcount=0.0
    totalcount=0.0
    for seq in seqs_dur:
        llimit = min(len(seq),l)
        for dur in seq[:l]:
            if dur % 6 == 0:
                validcount += 1
                break
        totalcount+=1
    return validcount/totalcount
def calcRM(seqs,seqs_dur,ns,ds,l=3,doDurs=False):
    ltuples = {}
    matchlist = []
    total = 0.0
    matches = 0.0
    for i in range(len(ns)):
        n = ns[i]
        d = ds[i]
        if len(n) < l:
            continue
        for j in range(len(n)-l+1):
            if doDurs:
                seq = (tuple(n[j:j+l]),tuple(d[j:j+l]))
            else:
                seq = tuple(n[j:j+l])
            ltuples[seq] = True
    for i in range(len(seqs)):
        n = seqs[i]
        d = seqs_dur[i]
        if len(n) < l:
            continue
        for j in range(len(n)-l+1):
            if doDurs:
                seq = (tuple(n[j:j+l]),tuple(d[j:j+l]))
            else:
                seq = tuple(n[j:j+l])
            try:
                result = ltuples[seq]
                if seq not in matchlist:
                    matchlist.append(seq)
                matches += 1
            except KeyError:
                pass
            total += 1
    print('Num types of matches:',len(matchlist))
    return matches/total
    
def calcPF(seqs):
    pitchcounts = {}
    for seq in seqs:
        for pitch in seq:
            if pitch not in pitchcounts.keys():
                pitchcounts[pitch] = 0
            pitchcounts[pitch] += 1
    return pitchcounts
def calcPFseq(seqs):
    totaldiffs = 0.0
    for seq in seqs:
        diffpitches = []
        for pitch in seq:
            if pitch not in diffpitches:
                diffpitches.append(pitch)
        totaldiffs += float(len(diffpitches))/len(seq)
    return totaldiffs/len(seqs)
def calcRF(act_seqs_durs):
    durcounts = {}
    for durseq in act_seqs_durs:
        for dur in durseq:
            if dur not in durcounts.keys():
                durcounts[dur] = 0
            durcounts[dur] += 1
    return durcounts
def calcRFseq(act_seqs_durs):
    totaldiffs = 0.0
    for durseq in act_seqs_durs:
        diffdurs = []
        for dur in durseq:
            if dur not in diffdurs:
                diffdurs.append(dur)
        totaldiffs += float(len(diffdurs))/len(durseq)
    return totaldiffs/len(act_seqs_durs)
def calcHC():
    pass

def getSeqs(fname,isDur=False,isTimesteps=False):
    with open(fname,'r') as gfile:
        unsups = json.load(gfile)
    seqs = [x[0] for x in unsups if x[0] != None]
    seqs_dur = [x[1] for x in unsups if x[1] != None]
    seqs_chordnotes = [x[2] for x in unsups if x[2] != None]
    seqs_chordkeys_temp = [x[4] for x in unsups if x[4] != None]
    n0s = [x[8] for x in unsups if x[5] != None]
    n1s = [x[7] for x in unsups if x[6] != None]
    n2s = [x[6] for x in unsups if x[7] != None]
    n3s = [x[5] for x in unsups if x[8] != None]
    seqs_startkeys = [x[0] for x in n0s]
    seqs_start_durs = [x[1] for x in n0s]
    seqs_startbeats = [x[2] for x in n0s]

    if isTimesteps:
        newseqs = []
        newdurseqs = []
        for seqindex in range(len(seqs)):
            seq = seqs[seqindex]
            prevpitch = seq[0]
            beatcount = 3
            newseq = []
            newdurseq = []
            for pitch in seq:
                if pitch == prevpitch:
                    beatcount = (beatcount+1)% 48
                if pitch != prevpitch:
                    newseq.append(prevpitch)
                    newdurseq.append(beatcount)
                    beatcount = (beatcount+1)% 48
                    prevpitch = pitch
            newseqs.append(newseq)
            newdurseqs.append(newdurseq)
        for n3 in n3s:
            n3[1] = 3
        for n2 in n2s:
            n2[1] = 2
        for n1 in n1s:
            n1[1] = 1
        for n0 in n0s:
            n0[1] = 0
        seqs_startkeys = [x[0] for x in n0s]
        seqs_start_durs = [x[1] for x in n0s]
        seqs_startbeats = [x[2] for x in n0s]
        seqs = newseqs
        seqs_dur = newdurseqs

    if isDur:
        actual_durs = seqs_dur
        for i in range(len(n1s)):
            n0 = n0s[i]
            n1 = n1s[i]
            n2 = n2s[i]
            n3 = n3s[i]
            n1[1] = (n0[1]+1+(n1[1]+1)) % 48
            n2[1] = (n1[1]+(n2[1]+1)) % 48
            n3[1] = (n2[1]+(n3[1]+1)) % 48
            n0[1] = (n0[1]+1) % 48
        seqs_start_durs = [x[1] for x in n0s]

        newdurseqs = []
        for i in range(len(seqs_dur)):
            durseq = seqs_dur[i]
            beat = n3s[i][1]
            newdurseq = []
            for dur in durseq:
                beat = (beat + (dur+1)) % 48
                newdurseq.append(beat)
            newdurseqs.append(newdurseq)
        seqs_dur = newdurseqs



    #seqs_startchordkeys = [(x[3],[1,0,0,0,1,0,0,1,0,0,0,0]) for x in n0s]

    seqs_chordkeys = []
    for seq in seqs_chordkeys_temp:
        newseq = []
        for key in seq:
            newseq.append(key)
        seqs_chordkeys.append(newseq)
    seqs_chords = []
    for i in range(len(seqs_chordnotes)):
        seq_chords = []
        seq_cnotes = seqs_chordnotes[i]
        seq_ckeys = seqs_chordkeys[i]
        for j in range(len(seq_cnotes)):
            notes = seq_cnotes[j]
            key = seq_ckeys[j] % 12
            seq_chords.append((key,notes))
        seqs_chords.append(seq_chords)
    if not isDur:
        actual_durs = calcDurs(seqs_dur,n0s)
    for dur in actual_durs:
        assert(dur != 0)
    return seqs,seqs_dur,seqs_chords,n0s,n1s,n2s,n3s,seqs_startkeys,seqs_start_durs,seqs_startbeats,actual_durs


def get_sequences_onefile(allpath,MAX_SEQ_DUR_LENGTH):
    MIDI_MAX = 108
    MIDI_MIN = 36
    REST_VAL = MIDI_MAX-MIDI_MIN+1
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

def writeLeadsheet(actseqs,actseqs_dur,actseqs_chords,midi_offset=0,rpitch=0,lsperdump=-1,fdir=0):
    melodyseqs = []
    for actseq in actseqs:
        for i in range(len(actseq)):
            if actseq[i] == rpitch:
                actseq[i] = None
            else:
                actseq[i] = actseq[i] + midi_offset
    for i in range(len(actseqs)):
        actseq = actseqs[i]
        actseq_dur = actseqs_dur[i]
        combinedseq = []
        for j in range(len(actseq)):
            combinedseq.append((actseq[j],actseq_dur[j]))
        melodyseqs.append(combinedseq)
    leadsheets = []
    for i in range(len(actseqs)):
        leadsheets.append(leadsheet.write_leadsheet(actseqs_chords[i], melodyseqs[i]))#, filename=fdir+str(i)+".ls")
        with open(fdir+str(i)+".ls",'w') as outfile:
            outfile.write(leadsheets[i])
    mergecount = 0
    chords = []
    melodies = []
    for i in range(len(leadsheets)):
        chords.extend(actseqs_chords[i])
        melodies.extend(melodyseqs[i])
        if ((lsperdump != -1) and (i+1) % lsperdump == 0) or i == len(leadsheets)-1:
            with open(fdir+"dump"+str(mergecount)+".ls",'w') as outfile:
                outfile.write(leadsheet.write_leadsheet(chords,melodies))
            chords = []
            melodies = []
            mergecount+=1

def generateLeadsheets(seqs,seqs_dur,seqs_chords,n0s,n1s,n2s,n3s,seqs_startkeys,seqs_start_durs,seqs_startbeats,
    USING_INTERVALS=False, REST_VAL=108-36+1, MIDI_OFFSET=36,OCT_THRESHOLD=72):
    actseqs = []
    actseqs_dur = []
    actseqs_chords = []
    oopsCount = 0
    for i in range(len(seqs_dur)):
        oct_change = 0#-12
        for note in seqs[i]:
            if note < OCT_THRESHOLD-MIDI_OFFSET:
                oct_change = 0
                break
        if n0s[i][0] < OCT_THRESHOLD-MIDI_OFFSET \
            or n1s[i][0] < OCT_THRESHOLD-MIDI_OFFSET \
            or n2s[i][0] < OCT_THRESHOLD-MIDI_OFFSET \
            or n3s[i][0] < OCT_THRESHOLD-MIDI_OFFSET:
            oct_change = 0
        actseq = []
        actseq_dur = []
        actseq_chords = []
        last_chords = []
        prev = seqs_startkeys[i]
        start_beat = 0#seqs_startbeats[i]
        prevdur = seqs_start_durs[i]
        durchange = (48+prevdur-start_beat)%48
        if durchange == 0:
            durchange = 48
        start_chord = seqs_chords[i][0]#seqs_startchordkeys[i]#seqs_chords[i][0]
        count = 0
        for _ in range(start_beat):
            actseq_chords.append(start_chord)
        if start_beat != 0:
            actseq.append(REST_VAL)
            actseq_dur.append(start_beat)
        actseq.append(prev+oct_change)
        actseq_dur.append(durchange)
        
        for _ in range(durchange):
            actseq_chords.append(start_chord)
            count += 1
        
        prev = n1s[i][0]
        durchange = (48+n1s[i][1]-prevdur)%48
        if durchange == 0:
            durchange = 48
        prevdur = n1s[i][1]
        if prev == REST_VAL:
            actseq.append(prev)
        else:
            actseq.append(prev+oct_change)
        actseq_dur.append(durchange)
        for _ in range(durchange):
            actseq_chords.append(start_chord)
            count += 1
            
        prev = n2s[i][0]
        durchange = (48+n2s[i][1]-prevdur)%48
        if durchange == 0:
            durchange = 48
        prevdur = n2s[i][1]
        if prev == REST_VAL:
            actseq.append(prev)
        else:
            actseq.append(prev+oct_change)
        actseq_dur.append(durchange)
        for _ in range(durchange):
            actseq_chords.append(start_chord)
            count += 1
            
        prev = n3s[i][0]
        durchange = (48+n3s[i][1]-prevdur)%48
        if durchange == 0:
            durchange = 48
        prevdur = n3s[i][1]
        if prev == REST_VAL:
            actseq.append(prev)
        else:
            actseq.append(prev+oct_change)
        actseq_dur.append(durchange)
        for _ in range(durchange):
            actseq_chords.append(start_chord)
            count += 1
        
        last_chord = None
        prevdur = n3s[i][1]
        totaldur = 0
        totaldur2 = 0
        pre_bp = 2
        td = 0
        for beatpos in seqs_dur[i]:
            dur = (beatpos-pre_bp-1+48) % 48 + 1
            td += dur
            pre_bp = beatpos
        #print(td,len(seqs_chords[i]))
        for j in range(len(seqs_dur[i])):
            isRest = False
            durenc = seqs_dur[i][j]
            noteval = seqs[i][j]
            #print(noteval-13)
            if USING_INTERVALS:
                noteval -= 13
                if noteval == 15:
                    noteval = REST_VAL
                    isRest = True
                if not isRest:
                    noteval = prev + noteval
                    prev = noteval
            dur = (48+durenc-prevdur-1)%48+1#+1#durdict[durenc]
            if dur == 0:
                dur = 48
            if totaldur+dur > len(seqs_chords[i]):
                dur = len(seqs_chords[i])-totaldur

            if dur == 0:
                #print("oops")
                oopsCount += 1
                #print(dur,totaldur,len(seqs_chords[i]))
                continue
            totaldur+=dur
            if j+1 < len(seqs_dur[i]):
                totaldur2+=dur
            prevdur = durenc
            if noteval == REST_VAL:
                actseq.append(noteval)
            else:
                actseq.append(noteval + oct_change)
            durcount = 0
            for _ in range(dur):
                #print("I: " + str(i) + ". Count: " + str(count))
                #actseq_chords.append(seqs_chords[i][j])
                #last_chord = seqs_chords[i][j]
                count+=1
                durcount+=1
                #if count >= 192:
                #    break
            actseq_dur.append(dur)
            #if count >= 192:
            #    break
        count = (start_beat+count) % 48
        diff = 48 - count
        if diff != 0:
            actseq.append(REST_VAL)
            actseq_dur.append(diff)
            for k in range(diff):
                #assert(last_chord != None)
                last_chords.append(seqs_chords[i][-1])
        # if i < 5:
        #     print("Totaldur: ",totaldur)
        #     print("Totaldur2: ",totaldur2)
        #     print("C len: ",len(seqs_chords[i]))
        actseqs.append(actseq)
        actseqs_dur.append(actseq_dur)
        actseqs_chords.append(actseq_chords+seqs_chords[i]+last_chords)
    print("Num Oops:",oopsCount)
    return actseqs,actseqs_dur,actseqs_chords
def calculateStats(seqs,seqs_dur,actual_durs,REST_VAL,doRM = False):
    print('QR',calcQR(actual_durs))
    print('CPR',calcCPR(seqs,2))
    print('RPD',calcRPD(seqs,actual_durs,24))
    print('TS',calcTS(seqs,12,REST_VAL))
    print('OR',calcOR(actual_durs,99))
    if doRM:
        MAX_SEQ_DUR_LENGTH = 48*4
        fname = "/home/nic/sequence_gan/parsed_leadsheets_bricked_all2_dur/pitchexpert_onehot_features.json"
        print(fname)
        n,d,_,_,_,starts = get_sequences_onefile(fname,MAX_SEQ_DUR_LENGTH)
        act_d = calcDurs_training(d,starts)
        for l in [3,4,5,6]:
            print('RM',calcRM(seqs,actual_durs,n,act_d,l=l,doDurs=False),'l',l)
    print('PF',calcPFseq(seqs))
    print('RF',calcRFseq(actual_durs))
    print('numSeqs',len(actual_durs))
    durtotal = 0
    for seq in actual_durs:
        for dur in seq:
            durtotal+=dur
    print('numBars',durtotal/48)

def parse_reals(seqs,seqs_dur,seqs_chords,highs,lows,starts):

    n0s = [x[3] for x in starts]
    n1s = [x[2] for x in starts]
    n2s = [x[1] for x in starts]
    n3s = [x[0] for x in starts]
    seqs_startkeys = [x[0] for x in n0s]
    seqs_start_durs = [x[1] for x in n0s]
    seqs_startbeats = [x[2] for x in n0s]


    return seqs,seqs_dur,seqs_chords,n0s,n1s,n2s,n3s,seqs_startkeys,seqs_start_durs,seqs_startbeats

def main():
    USING_GENS = True
    USING_INTERVALS=False
    USING_REALS = False
    USING_MAGENTA = False
    REST_VAL=108-36+1
    MIDI_OFFSET=36
    OCT_THRESHOLD=72
    if USING_GENS:
        isTimesteps = True
        isDur = False
        fname = './gens_g9.json'#'./deployer_gens_beatpos.json'#'/home/nic/sequence_gan/zbeatposition_stuff/gens_g99.json'
        fdir = '/home/nic/0_beatpos3/'
        if isDur:
            fname = '/home/nic/sequence_gan/zduration_stuff/gens_g99.json'
            fdir = '/home/nic/0_duration2/'
        elif isTimesteps:
            fname = '/home/nic/sequence_gan/ztimestep_stuff/tgens_timestep_g30.json'#27.json'
            fdir = '/home/nic/0_timestep2/'
        doRM = True
        print(fname)
        seqs,seqs_dur,seqs_chords,n0s,n1s,n2s,n3s,seqs_startkeys,seqs_start_durs,seqs_startbeats,actual_durs = getSeqs(fname,isTimesteps=isTimesteps,isDur=isDur)
        print("Num gen seqs:",len(seqs))
        actseqs,actseqs_dur,actseqs_chords = generateLeadsheets(seqs,seqs_dur,seqs_chords,n0s,n1s,n2s,n3s,seqs_startkeys,seqs_start_durs,seqs_startbeats)
        print("Num act seqs:", len(actseqs))
        for the_file in os.listdir(fdir):
            file_path = os.path.join(fdir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        lss = writeLeadsheet(actseqs,actseqs_dur,actseqs_chords,midi_offset=MIDI_OFFSET,rpitch=REST_VAL,lsperdump=-1,fdir = fdir)
        #calculateStats(seqs,seqs_dur,actual_durs,REST_VAL,doRM=doRM)
    elif USING_REALS:
        MAX_SEQ_DUR_LENGTH = 48*4
        fname = "/home/nic/sequence_gan/parsed_leadsheets_bricked_all2_dur/pitchexpert_onehot_features.json"
        print(fname)
        seqs,seqs_dur,seqs_chords,highs,lows,starts = get_sequences_onefile(fname,MAX_SEQ_DUR_LENGTH)
        seqs,seqs_dur,seqs_chords,n0s,n1s,n2s,n3s,seqs_startkeys,seqs_start_durs,seqs_startbeats = parse_reals(seqs,seqs_dur,seqs_chords,highs,lows,starts)
        actseqs,actseqs_dur,actseqs_chords = generateLeadsheets(seqs,seqs_dur,seqs_chords,n0s,n1s,n2s,n3s,seqs_startkeys,seqs_start_durs,seqs_startbeats)
        actual_durs = calcDurs_training(seqs_dur,starts)
        #calculateStats(seqs,seqs_dur,actual_durs,REST_VAL)
        if True:
            fdir = '/home/nic/0_corpus/'
            for the_file in os.listdir(fdir):
                file_path = os.path.join(fdir, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
            lss = writeLeadsheet(actseqs,actseqs_dur,actseqs_chords,midi_offset=MIDI_OFFSET,rpitch=REST_VAL,lsperdump=-1,fdir = fdir)

    elif USING_MAGENTA:
        fname = "/home/nic/sequence_gan/mag_gens.json"
        with open(fname,'r') as infile:
            data = json.load(infile)
        seqs = data['seqs']
        seqs_dur = data['seqs_dur']
        seqs_act_durs = data['seqs_act_durs']
        seqs_chords = data['seqs_chords']
        tenth_index = int(len(seqs)/10)
        #calculateStats(seqs,seqs_dur,seqs_act_durs,REST_VAL)
        if True:
            for i in range(10):
                fdir = '/home/nic/0_mag_'+str(i)+'/'
                if not os.path.exists(fdir):
                    os.makedirs(fdir)
                for the_file in os.listdir(fdir):
                    file_path = os.path.join(fdir, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)
                lss = writeLeadsheet(seqs[tenth_index*i:min(tenth_index*(i+1),len(seqs))],
                    seqs_act_durs[tenth_index*i:min(tenth_index*(i+1),len(seqs))],
                    seqs_chords[tenth_index*i:min(tenth_index*(i+1),len(seqs))],
                    midi_offset=MIDI_OFFSET,rpitch=REST_VAL,lsperdump=-1,fdir = fdir)

if __name__ == '__main__':
    main()