import xmltodict
import deployer
import os
import json

def main():
	folder = '/home/nic/Music/mag_gens/xmls/'
	seqs = []
	seqs_dur = []
	seqs_act_durs = []
	seqs_chords = []
	MIDI_VALS = {'C': 0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
	ORNAMENT_ADJUST = {'flat':-1,'sharp':1,'double-sharp':2,'double-flat':-2,'flat-flat':-2}
	WHOLE_TIMESTEPS = 48
	quarter_timesteps = WHOLE_TIMESTEPS/4
	REST_VAL = deployer.REST_VAL
	filecount = -1
	for the_file in os.listdir(folder):
		filecount += 1
		if filecount % 1000 == 0:
			print(filecount)
		fname = os.path.join(folder, the_file)
		with open(fname,'r') as infile:
			doc = xmltodict.parse(infile.read())

		measures = doc['score-partwise']['part'][0]['measure'] # Get piano part
		if not isinstance(measures,list):
			measures = [measures]
		
		divisions = measures[0]['attributes']
		if isinstance(divisions,list):
			divisions = int(divisions[0]['divisions'])
		else:
			divisions = int(divisions['divisions'])# Get how many times a quarter note is segmented
		division_timesteps = quarter_timesteps/divisions

		seq = []
		seq_dur = []
		seq_act_durs = []
		beatpos = 0
		for i in range(len(measures)):
			measure = measures[i]
			notes = measure['note']
			if not isinstance(notes,list):
				notes = [notes]
			#recorded_accidentals = {}
			for j in range(len(notes)):
				note = notes[j]
				if 'rest' in note.keys():
					pitch_val = REST_VAL
				else:
					pitch_step = note['pitch']['step']
					pitch_octave = int(note['pitch']['octave'])
					pitch_val = MIDI_VALS[pitch_step]+12*(pitch_octave+1)
					if 'accidental' in note.keys():
						accidental = note['accidental']
						if accidental != 'natural':
							pitch_val += ORNAMENT_ADJUST[accidental]
					pitch_val -= deployer.MIDI_MIN
					assert(pitch_val >= 0)
				duration_val = int(note['duration'])*division_timesteps
				beatpos = (beatpos + duration_val) % WHOLE_TIMESTEPS

				seq.append(pitch_val)
				seq_dur.append(beatpos)
				seq_act_durs.append(duration_val)
		seqs.append(seq)
		seqs_dur.append(seq_dur)
		seqs_act_durs.append(seq_act_durs)

		##### NOW PARSE CHORDS
		measures = doc['score-partwise']['part'][1]['measure'] # Get piano part
		if not isinstance(measures,list):
			measures = [measures]
		
		divisions = measures[0]['attributes']
		if isinstance(divisions,list):
			divisions = int(divisions[0]['divisions'])
		else:
			divisions = int(divisions['divisions'])# Get how many times a quarter note is segmented
		division_timesteps = quarter_timesteps/divisions

		seq_chords = []
		for i in range(len(measures)):
			measure = measures[i]
			notes = measure['note']
			if not isinstance(notes,list):
				notes = [notes]
			curr_x = float(notes[0]['@default-x'])
			curr_y = 999#notes[0]['@default-y']
			curr_ckey = -1
			curr_cnotes = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
			curr_dur = int(notes[0]['duration'])*division_timesteps
			#recorded_accidentals = {}
			for j in range(len(notes)):
				note = notes[j]
				isRest = False
				if 'rest' in note.keys():
					pitch_val = 999
					curr_cnotes = [0,0,0,0,0,0,0,0,0,0,0,0]
					isRest = True
				else:
					notex = float(note['@default-x'])
					notey = float(note['@default-y'])
					pitch_step = note['pitch']['step']
					pitch_octave = int(note['pitch']['octave'])
					pitch_val = MIDI_VALS[pitch_step]+12*(pitch_octave+1)
					if 'accidental' in note.keys():
						accidental = note['accidental']
						if accidental != 'natural':
							pitch_val += ORNAMENT_ADJUST[accidental]
					pitch_val = pitch_val % 12
				if not isRest and abs(notex - curr_x)<0.001:
					if notey < curr_y:
						curr_y = notey
						curr_ckey = pitch_val
				duration_val = int(note['duration'])*division_timesteps
				if (not isRest and abs(notex - curr_x)>0.001) or j == len(notes)-1:
					# Half-bar chord or end of measure
					for _ in range(curr_dur):
						seq_chords.append([curr_ckey,curr_cnotes])
					curr_ckey = pitch_val
					curr_cnotes = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
		totaldur = sum(seq_act_durs)
		endrest = totaldur-len(seq_chords)
		for _ in range(endrest):
			seq_chords.append([0,[0,0,0,0,0,0,0,0,0,0,0,0]])
		seqs_chords.append(seq_chords)
		"""print(seq)
		print()
		print(seq_dur)
		print()
		print(seq_act_durs)
		print()
		print([x[0] for x in seq_chords])
		print()
		print(len(seq_chords)/48.0)
		print(totaldur)"""
	datadump = {"seqs":seqs,"seqs_dur":seqs_dur,"seqs_act_durs":seqs_act_durs,"seqs_chords":seqs_chords}
	print("Dumping...")
	with open("./mag_gens.json",'w') as outfile:
		json.dump(datadump,outfile)

if __name__ == '__main__':
    main()