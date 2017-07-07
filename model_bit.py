__doc__ = """RNN-based GAN.  For applying Generative Adversarial Networks to sequential data."""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


def _cumsum(x, length):
    lower_triangular_ones = tf.constant(
        np.tril(np.ones((length, length))),
        dtype=tf.float32)
    return tf.reshape(
            tf.matmul(lower_triangular_ones,
                      tf.reshape(x, [length, 1])),
            [length])


def _backwards_cumsum(x, length):
    duple_length = tf.stack([length,length])
    length_one = tf.stack([length,tf.constant(1,dtype=tf.int32)])
    upper_triangular_ones = tf.matrix_band_part(tf.ones(duple_length), 0, -1)
    mult = tf.matmul(upper_triangular_ones,
                      tf.reshape(x, length_one))
    return tf.reshape(mult,[length])


WHOLE = tf.constant(0,dtype=tf.int32)
WHOLE_DOTTED = tf.constant(1,dtype=tf.int32)
WHOLE_TRIPLET = tf.constant(2,dtype=tf.int32)
HALF = tf.constant(3,dtype=tf.int32)
HALF_DOTTED = tf.constant(4,dtype=tf.int32)
HALF_TRIPLET = tf.constant(5,dtype=tf.int32)
QUARTER = tf.constant(6,dtype=tf.int32)
QUARTER_DOTTED = tf.constant(7,dtype=tf.int32)
QUARTER_TRIPLET = tf.constant(8,dtype=tf.int32)
EIGHTH = tf.constant(9,dtype=tf.int32)
EIGHTH_DOTTED = tf.constant(10,dtype=tf.int32)
EIGHTH_TRIPLET = tf.constant(11,dtype=tf.int32)
SIXTEENTH = tf.constant(12,dtype=tf.int32)
SIXTEENTH_DOTTED = tf.constant(13,dtype=tf.int32)
SIXTEENTH_TRIPLET = tf.constant(14,dtype=tf.int32)

durdict_ints = {
    WHOLE: 48,
    WHOLE_DOTTED: 72,
    WHOLE_TRIPLET: 16,
    HALF: 24,
    HALF_DOTTED: 36,
    HALF_TRIPLET: 8,
    QUARTER: 12,
    QUARTER_DOTTED: 18,
    QUARTER_TRIPLET: 4,
    EIGHTH: 6,
    EIGHTH_DOTTED: 9,
    EIGHTH_TRIPLET: 2,
    SIXTEENTH: 3,
    SIXTEENTH_DOTTED: 4,
    SIXTEENTH_TRIPLET: 1
}
durdict = {
    WHOLE: tf.constant(48,dtype=tf.int32),
    WHOLE_DOTTED: tf.constant(72,dtype=tf.int32),
    WHOLE_TRIPLET: tf.constant(16,dtype=tf.int32),
    HALF: tf.constant(24,dtype=tf.int32),
    HALF_DOTTED: tf.constant(36,dtype=tf.int32),
    HALF_TRIPLET: tf.constant(8,dtype=tf.int32),
    QUARTER: tf.constant(12,dtype=tf.int32),
    QUARTER_DOTTED: tf.constant(18,dtype=tf.int32),
    QUARTER_TRIPLET: tf.constant(4,dtype=tf.int32),
    EIGHTH: tf.constant(6,dtype=tf.int32),
    EIGHTH_DOTTED: tf.constant(9,dtype=tf.int32),
    EIGHTH_TRIPLET: tf.constant(2,dtype=tf.int32),
    SIXTEENTH: tf.constant(3,dtype=tf.int32),
    SIXTEENTH_DOTTED: tf.constant(4,dtype=tf.int32),
    SIXTEENTH_TRIPLET: tf.constant(1,dtype=tf.int32)
}

ONE_HOT = 0
BITS = 1
IMPROVISOR_WHOLE_NOTE_DURATION = 48.0
WHOLE_NOTE_DURATION = 128.0

class RNN(object):

    def __init__(self, num_emb, num_emb_dur, emb_dim, emb_dim_dur, hidden_dim, hidden_dim_dur,num_hidden_layers,
                 max_sequence_length, start_token, start_token_dur, start_token_pos_low, start_token_pos_high,
                 learning_rate=0.01, reward_gamma=0.9,MIDI_MIN=55,MIDI_MAX=89,ENCODING=ONE_HOT):
        powers_of_two = tf.constant([1,2,4,8,16,32,64,128])
        def duration_tensor_to_beat_duration(dtensor):
            return tf.constant(0,dtype=tf.int32)
        """if ENCODING == ONE_HOT:
            def duration_tensor_to_beat_duration(dtensor):
                return tf.case([
                    (tf.equal(dtensor,WHOLE), lambda:durdict[WHOLE]),
                    (tf.equal(dtensor,WHOLE_DOTTED), lambda:durdict[WHOLE_DOTTED]),
                    (tf.equal(dtensor,WHOLE_TRIPLET), lambda:durdict[WHOLE_TRIPLET]),
                    (tf.equal(dtensor,HALF), lambda:durdict[HALF]),
                    (tf.equal(dtensor,HALF_DOTTED), lambda:durdict[HALF_DOTTED]),
                    (tf.equal(dtensor,HALF_TRIPLET), lambda:durdict[HALF_TRIPLET]),
                    (tf.equal(dtensor,QUARTER), lambda:durdict[QUARTER]),
                    (tf.equal(dtensor,QUARTER_DOTTED), lambda:durdict[QUARTER_DOTTED]),
                    (tf.equal(dtensor,QUARTER_TRIPLET), lambda:durdict[QUARTER_TRIPLET]),
                    (tf.equal(dtensor,EIGHTH), lambda:durdict[EIGHTH]),
                    (tf.equal(dtensor,EIGHTH_DOTTED), lambda:durdict[EIGHTH_DOTTED]),
                    (tf.equal(dtensor,EIGHTH_TRIPLET), lambda:durdict[EIGHTH_TRIPLET]),
                    (tf.equal(dtensor,SIXTEENTH), lambda:durdict[SIXTEENTH]),
                    (tf.equal(dtensor,SIXTEENTH_DOTTED), lambda:durdict[SIXTEENTH_DOTTED]),
                    (tf.equal(dtensor,SIXTEENTH_TRIPLET), lambda:durdict[SIXTEENTH_TRIPLET]),],
                    default=lambda: tf.constant(-1,dtype=tf.int32))
        elif ENCODING == BITS:
            def duration_tensor_to_beat_duration(dtensor):
                return tf.to_int32(tf.to_float(dtensor)*IMPROVISOR_WHOLE_NOTE_DURATION/WHOLE_NOTE_DURATION)"""

        self.num_emb = num_emb
        self.num_emb_dur = num_emb_dur
        self.emb_dim = emb_dim
        self.emb_dim_dur = emb_dim_dur
        self.hidden_dim = hidden_dim
        self.hidden_dim_dur = hidden_dim_dur
        self.num_hidden_layers = num_hidden_layers
        self.max_sequence_length = max_sequence_length
        self.sequence_length = tf.placeholder(tf.int32,shape=[]) #sequence_length
        self.start_token = tf.constant(start_token, dtype=tf.int32)
        self.start_token_dur = tf.constant(start_token_dur, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []
        self.MIDI_MIN = MIDI_MIN # lowest note value found in trainingset
        self.MIDI_MAX = MIDI_MAX # highest note value found in trainingset
        self.REST_VAL = self.MIDI_MAX-self.MIDI_MIN+1

        self.expected_reward = tf.Variable(tf.zeros([self.max_sequence_length]))

        numBeatsInMeasure = tf.constant(48,dtype=tf.int32)
        beatsConsideredVec = tf.constant([durdict_ints[WHOLE],durdict_ints[HALF],durdict_ints[QUARTER],durdict_ints[EIGHTH],durdict_ints[SIXTEENTH],\
            durdict_ints[HALF_TRIPLET],durdict_ints[QUARTER_TRIPLET],durdict_ints[EIGHTH_TRIPLET]])
        #beatsConsideredVec = tf.constant([WHOLE_NOTE,HALF_NOTE,QUARTER_NOTE,EIGHTH_NOTE,SIXTEENTH_NOTE,HALF_TRIPLET,QUARTER_TRIPLET,EIGHTH_TRIPLET])
        self.lenBeatVec = tf.size(beatsConsideredVec)

        with tf.variable_scope('generator'):
            # Embedding matrix for notes
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)

            # Embedding matrix for dur bit
            self.g_embeddings_dur = tf.Variable(self.init_matrix([self.num_emb_dur,self.emb_dim_dur]))
            self.g_params.append(self.g_embeddings_dur)

            self.g_recurrent_unit = self.create_recurrent_unit_pitch(self.emb_dim,self.emb_dim_dur, self.hidden_dim, self.g_params)  # maps h_tm1 to h_t for generator
            #self.g_recurrent_unit_dur = self.create_recurrent_unit_dur(self.emb_dim_dur, self.hidden_dim_dur, self.g_params)


            self.g_hidden = self.create_recurrent_unit_hidden(self.hidden_dim,self.g_params)
            self.g_hidden_dur = self.create_recurrent_unit_hidden(self.hidden_dim,self.g_params)
            #self.g_hidden_layers = []
            #self.g_hidden_layers_dur = []
            #for _ in range(self.num_hidden_layers):
            #    self.g_hidden_layers.append(self.create_recurrent_unit_hidden(self.hidden_dim,self.g_params))
            #    self.g_hidden_layers_dur.append(self.create_recurrent_unit_hidden(self.hidden_dim_dur,self.g_params))

            self.g_output_unit = self.create_output_unit(self.num_emb, self.emb_dim, self.hidden_dim, self.g_params, self.g_embeddings)  # maps h_t to o_t (output token logits)
            

            
            self.g_output_unit_b1 = self.create_output_unit_no_emb(2,self.hidden_dim,self.g_params)
            self.g_output_unit_b2 = self.create_output_unit_no_emb(2,self.hidden_dim,self.g_params)
            self.g_output_unit_b3 = self.create_output_unit_no_emb(2,self.hidden_dim,self.g_params)
            self.g_output_unit_b4 = self.create_output_unit_no_emb(2,self.hidden_dim,self.g_params)
            self.g_output_unit_b5 = self.create_output_unit_no_emb(2,self.hidden_dim,self.g_params)
            self.g_output_unit_b6 = self.create_output_unit_no_emb(2,self.hidden_dim,self.g_params)
            self.g_output_unit_b7 = self.create_output_unit_no_emb(2,self.hidden_dim,self.g_params)
            
            #self.g_output_unit_dur = self.create_output_unit(self.num_emb_dur, self.emb_dim_dur, self.hidden_dim, self.g_params, self.g_embeddings_dur)

        with tf.variable_scope('discriminator'):
            # Embedding matrix for notes
            self.d_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.d_params.append(self.d_embeddings)

            # Embedding matrix for dur bit
            self.d_embeddings_dur = tf.Variable(self.init_matrix([self.num_emb_dur, self.emb_dim_dur]))
            self.d_params.append(self.d_embeddings_dur)

            self.d_recurrent_unit = self.create_recurrent_unit_pitch(self.emb_dim,self.emb_dim_dur, self.hidden_dim, self.d_params)  # maps h_tm1 to h_t for discriminator
            #self.d_recurrent_unit_dur = self.create_recurrent_unit_dur(self.emb_dim_dur, self.hidden_dim_dur, self.d_params)  # maps h_tm1 to h_t for discriminator
            self.d_classifier_unit = self.create_classifier_unit(self.d_params)  # maps h_t to class prediction logits
            self.d_h0 = tf.Variable(self.init_vector([self.hidden_dim]))
            self.d_h0_dur = tf.Variable(self.init_vector([self.hidden_dim_dur]))
            self.d_params.append(self.d_h0)
            self.d_params.append(self.d_h0_dur)

        self.h0 = tf.placeholder(tf.float32, shape=[self.hidden_dim])  # initial random vector for generator
        self.h0_dur = tf.placeholder(tf.float32, shape=[self.hidden_dim_dur]) # initial random vector for generator dur
        self.h0s = tf.placeholder(tf.float32, shape=[self.num_hidden_layers, self.hidden_dim])  # initial random vector for generator
        self.h0s_dur = tf.placeholder(tf.float32, shape=[self.num_hidden_layers, self.hidden_dim_dur]) # initial random vector for generator dur
        self.temph0s = tf.placeholder(tf.float32, shape=[self.hidden_dim])
        self.temph0s_dur = tf.placeholder(tf.float32, shape=[self.hidden_dim_dur])
        self.x = tf.placeholder(tf.int32, shape=[None])  # sequence of indices of true note intervals, not including start token
        self.x_dur = tf.placeholder(tf.int32, shape=[None,7])  # sequence of indices of true durs, not including start token
        self.x_pitch = tf.placeholder(tf.int32,shape=[None]) # sequence of indices of true notes, not including start token
        self.samples = tf.placeholder(tf.float32, shape=[None])  # random samples from [0, 1]
        self.chordKeys = tf.placeholder(tf.int32, shape=[None]) # sequence of chord key values
        self.start_pitch = tf.placeholder(tf.int32,shape=[]) # starting pitch for INTERVALs
        self.start_duration = tf.placeholder(tf.int32,shape=[7]) # starting duration
        self.chordKeys_onehot = tf.placeholder(tf.int32, shape=[None]) # sequence of chord keys, onehot
        self.chordNotes = tf.placeholder(tf.int32, shape=[None, 12]) # sequence of vectors of notes in the chord
        self.lows = tf.placeholder(tf.float32, shape=[None]) # sequence of low pos ratios
        self.highs = tf.placeholder(tf.float32, shape=[None]) # sequence of high pos ratios

        # generator on initial randomness
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_o_dur = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x_dur = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_high = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_low = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        samples = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        samples = samples.unstack(self.samples)

        def _g_hidden_layers(j,hidden_dim,x_t,h_tm1s,layers):
            gru_layer = layers[j]
            h_t = gru_layer(hidden_dim, x_t, h_tm1s[j])
            h_tm1s[j] = h_t
            return j+1,hidden_dim, x_t,h_tm1s,layers

        def _g_recurrence(i, beat_count, pitch_count,
            chordkey_vec, chordnote_vec, low, high,
            a_count, prev_a,
            rep_count, prev_token, 
            x_t,b_t, h_tm1,h_tm1s, h_tm1_dur,h_tm1s_dur,
            gen_o, gen_x, gen_o_dur, gen_x_dur, gen_low, gen_high):   


            beat = tf.mod(beat_count,numBeatsInMeasure)
            beatVec = tf.map_fn(lambda i : tf.to_float(tf.equal(tf.mod(beat,i),tf.constant(0,dtype=tf.int32))), beatsConsideredVec, dtype=tf.float32)
            sample = samples.read(i)

            # Feed pitch inputs to input GRU layer of pitch RNN
            h_t = self.g_recurrent_unit(self.emb_dim, self.hidden_dim, x_t, b_t, beatVec, rep_count, h_tm1,chordkey_vec,chordnote_vec,low,high)

            # Feed output to softmax unit to get next predicted token
            ob1_t = self.g_output_unit_b1(h_t)
            ob2_t = self.g_output_unit_b2(h_t)
            ob3_t = self.g_output_unit_b3(h_t)
            ob4_t = self.g_output_unit_b4(h_t)
            ob5_t = self.g_output_unit_b5(h_t)
            ob6_t = self.g_output_unit_b6(h_t)
            ob7_t = self.g_output_unit_b7(h_t)
            
            ob1_cumsum = _cumsum(ob1_t, 2)
            ob2_cumsum = _cumsum(ob2_t, 2)
            ob3_cumsum = _cumsum(ob3_t, 2)
            ob4_cumsum = _cumsum(ob4_t, 2)
            ob5_cumsum = _cumsum(ob5_t, 2)
            ob6_cumsum = _cumsum(ob6_t, 2)
            ob7_cumsum = _cumsum(ob7_t, 2)
            next_token_b1 = tf.to_int32(tf.reduce_min(tf.where(sample < ob1_cumsum)))
            next_token_b2 = tf.to_int32(tf.reduce_min(tf.where(sample < ob2_cumsum)))
            next_token_b3 = tf.to_int32(tf.reduce_min(tf.where(sample < ob3_cumsum)))
            next_token_b4 = tf.to_int32(tf.reduce_min(tf.where(sample < ob4_cumsum)))
            next_token_b5 = tf.to_int32(tf.reduce_min(tf.where(sample < ob5_cumsum)))
            next_token_b6 = tf.to_int32(tf.reduce_min(tf.where(sample < ob6_cumsum)))
            next_token_b7 = tf.to_int32(tf.reduce_min(tf.where(sample < ob7_cumsum)))
            b_t = tf.stack([next_token_b1,next_token_b2,next_token_b3,next_token_b4,next_token_b5,next_token_b6,next_token_b7])

            ob1_prob = tf.gather(ob1_t,next_token_b1)
            ob2_prob = tf.gather(ob2_t,next_token_b2)
            ob3_prob = tf.gather(ob3_t,next_token_b3)
            ob4_prob = tf.gather(ob4_t,next_token_b4)
            ob5_prob = tf.gather(ob5_t,next_token_b5)
            ob6_prob = tf.gather(ob6_t,next_token_b6)
            ob7_prob = tf.gather(ob7_t,next_token_b7)
            

            # Feed output to softmax unit to get next predicted token
            o_t = self.g_output_unit(self.g_embeddings, self.num_emb, self.hidden_dim, h_t)
            o_cumsum = _cumsum(o_t, self.num_emb)  # prepare for sampling
            next_token = tf.to_int32(tf.reduce_min(tf.where(sample < o_cumsum)))   # sample
            o_prob = tf.gather(o_t,next_token)

            all_prob = tf.stack([ob1_prob,ob2_prob,ob3_prob,ob4_prob,ob5_prob,ob6_prob,ob7_prob,o_prob])
            mean_prob = tf.reduce_mean(all_prob)

            # Calculate low and high for next note. COMPUTATION DEPENDS ON INTERVAL USAGE.
            newLow = tf.cond(tf.equal(next_token,tf.constant(self.REST_VAL,dtype=tf.int32)), lambda: tf.constant(0.0,dtype=tf.float32), lambda: tf.to_float(next_token)/tf.constant((MIDI_MAX- MIDI_MIN),dtype=tf.float32))
            newHigh = tf.cond(tf.equal(next_token,tf.constant(self.REST_VAL,dtype=tf.int32)), lambda: tf.constant(0.0,dtype=tf.float32), lambda: tf.constant(1.0,dtype=tf.float32)-newLow)
            newPitch = tf.cond(tf.equal(next_token,tf.constant(self.REST_VAL,dtype=tf.int32)), lambda: pitch_count, lambda: pitch_count+next_token+self.MIDI_MIN )

            gen_low = gen_low.write(i, newLow)
            gen_high = gen_high.write(i, newHigh)

            x_tp1 = tf.gather(self.g_embeddings, next_token)
            gen_o = gen_o.write(i, mean_prob)  # we only need the sampled token's probability
            gen_x = gen_x.write(i, next_token)  # indices, not embeddings

            gen_x_dur = gen_x_dur.write(i, b_t)
            dur = duration_tensor_to_beat_duration(b_t)

            return i + 1, beat+dur, newPitch,\
                self.chordKeys_onehot[(i+dur) % self.sequence_length],self.chordNotes[(i+dur) % self.sequence_length],newLow,newHigh,\
                tf.constant(0,dtype=tf.int32),tf.constant(0,dtype=tf.int32),\
                tf.multiply(rep_count,tf.to_int32(tf.equal(prev_token,next_token)))+1,next_token, \
                x_tp1, b_t, h_t, h_tm1s, h_tm1_dur, h_tm1s_dur,\
                gen_o, gen_x,gen_o_dur,gen_x_dur, gen_low, gen_high

        _, _, _, \
        _, _, _, _, \
        _, _, \
        _, _, \
        _, _, _, _, _, _, \
        self.gen_o, self.gen_x, self.gen_o_dur, self.gen_x_dur, self.gen_low, self.gen_high = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20,_21,_22,_23,_24: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),duration_tensor_to_beat_duration(self.start_duration),self.start_pitch,
                self.chordKeys_onehot[0],self.chordNotes[0],tf.constant(0.0,dtype=tf.float32),tf.constant(0.0,dtype=tf.float32),
                tf.constant(0,dtype=tf.int32),tf.constant(0,dtype=tf.int32),
                tf.constant(0,dtype=tf.int32),self.start_pitch,
                tf.gather(self.g_embeddings, self.start_pitch),self.start_duration,self.h0,self.h0s,self.h0_dur,self.h0s_dur, 
                gen_o, gen_x,gen_o_dur,gen_x_dur,gen_low, gen_high))
        
        # discriminator on generated and real data: note vars
        d_gen_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)
        d_real_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        self.gen_x = self.gen_x.stack()
        emb_gen_x = tf.gather(self.d_embeddings, self.gen_x)
        ta_emb_gen_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_gen_x = ta_emb_gen_x.unstack(emb_gen_x)

        emb_real_x = tf.gather(self.d_embeddings, self.x)
        ta_emb_real_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_real_x = ta_emb_real_x.unstack(emb_real_x)

        # discriminator on generated and real data: dur vars

        #emb_real_x_dur = tf.gather(self.d_embeddings_dur, self.x_dur)
        ta_emb_real_x_dur = tensor_array_ops.TensorArray(
            dtype=tf.int32, size=self.sequence_length,infer_shape=True)
        ta_emb_real_x_dur = ta_emb_real_x_dur.unstack(self.x_dur)
        #self.gen_x_dur = tf.stack(self.gen_x_dur)

        # discriminator recurrence

        def _d_recurrence(i,beat_count,
            chordkey_vec, chordnote_vec, inputs_lows, inputs_highs,           
            a_count, prev_a, durs,
            rep_count, prev_token, notes, 
            inputs, h_tm1, h_tm1_dur, pred):
            low = inputs_lows[i]
            high = inputs_highs[i]
            beat = tf.mod(beat_count,numBeatsInMeasure)
            beatVec = tf.map_fn(lambda i : tf.to_float(tf.equal(tf.mod(beat,i),tf.constant(0,dtype=tf.int32))), beatsConsideredVec, dtype=tf.float32)
            x_t = inputs.read(i)
            next_token = notes[i]
            #a_t = inputs_dur.read(i)
            next_a = durs.read(i)
            h_t = self.d_recurrent_unit(self.emb_dim, self.hidden_dim, x_t, next_a, beatVec, rep_count, h_tm1,chordkey_vec,chordnote_vec,low,high)
            #h_t_dur = self.d_recurrent_unit_dur(self.emb_dim_dur, self.hidden_dim_dur, a_t,beatVec, a_count, h_tm1_dur)
            y_t = self.d_classifier_unit(h_t)
            pred = pred.write(i, y_t)
            dur = duration_tensor_to_beat_duration(next_a)
            return i + 1, beat+dur,\
                self.chordKeys_onehot[(i+dur) % self.sequence_length],self.chordNotes[(i+dur) % self.sequence_length],inputs_lows,inputs_highs,\
                tf.constant(0,dtype=tf.int32), next_a,durs,\
                tf.multiply(rep_count,tf.to_int32(tf.equal(prev_token,next_token)))+1, next_token,notes,\
                inputs, h_t,h_tm1_dur, pred

        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, self.d_gen_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15: i < self.sequence_length,
            body=_d_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),duration_tensor_to_beat_duration(self.start_duration),
                self.chordKeys_onehot[0],self.chordNotes[0],self.gen_low.stack(),self.gen_high.stack(),
                tf.constant(0,dtype=tf.int32), self.start_duration, self.gen_x_dur,
                tf.constant(0,dtype=tf.int32), self.start_pitch, self.gen_x,
                ta_emb_gen_x,self.d_h0,self.d_h0_dur, d_gen_predictions))
        self.d_gen_predictions = tf.reshape(
                self.d_gen_predictions.stack(),
                [self.sequence_length])


        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, self.d_real_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15: i < self.sequence_length,
            body=_d_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),duration_tensor_to_beat_duration(self.start_duration),
                self.chordKeys_onehot[0],self.chordNotes[0],self.lows,self.highs,
                tf.constant(0,dtype=tf.int32),self.start_duration,ta_emb_real_x_dur,
                tf.constant(0,dtype=tf.int32),self.start_pitch,self.x,
                ta_emb_real_x,self.d_h0,self.d_h0_dur, d_real_predictions))
        self.d_real_predictions = tf.reshape(
                self.d_real_predictions.stack(),
                [self.sequence_length])

        # supervised pretraining for generator: note vars
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        emb_x = tf.gather(self.g_embeddings, self.x)
        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(emb_x)

        # supervised pretraining for generator: dur vars
        g_predictions_dur = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        #emb_x_dur = tf.gather(self.g_embeddings_dur, self.x_dur)
        ta_emb_x_dur = tensor_array_ops.TensorArray(
            dtype=tf.int32, size=self.sequence_length, infer_shape=True)
        ta_emb_x_dur = ta_emb_x_dur.unstack(self.x_dur)

        # pretrain recurrence
        
        def _pretrain_recurrence(i, beat_count,
            chordkey_vec, chordnote_vec, low, high,      
            a_count,prev_a,
            rep_count, prev_token, 
            x_t, a_t, h_tm1,
            g_predictions, g_predictions_dur):
            beat = tf.mod(beat_count,numBeatsInMeasure)
            beatVec = tf.map_fn(lambda i : tf.to_float(tf.equal(tf.mod(beat,i),tf.constant(0,dtype=tf.int32))), beatsConsideredVec, dtype=tf.float32)
            firstH = self.g_recurrent_unit(self.emb_dim, self.hidden_dim, x_t, a_t, beatVec, rep_count, h_tm1,chordkey_vec,chordnote_vec,low,high)
            #secondH = self.g_hidden(self.hidden_dim, firstH, h_tm1s)
            #firstH_dur = self.g_recurrent_unit_dur(self.emb_dim_dur, self.hidden_dim_dur, a_t, beatVec, a_count, h_tm1_dur)
            #secondH_dur = self.g_hidden_dur(self.hidden_dim, firstH, h_tm1s)

            o_t = self.g_output_unit(self.g_embeddings, self.num_emb, self.hidden_dim, firstH)
            #oa_t = self.g_output_unit_dur(self.g_embeddings_dur, self.num_emb_dur,self.hidden_dim, secondH_dur)
            
            ob1_t = self.g_output_unit_b1(firstH)
            ob2_t = self.g_output_unit_b2(firstH)
            ob3_t = self.g_output_unit_b3(firstH)
            ob4_t = self.g_output_unit_b4(firstH)
            ob5_t = self.g_output_unit_b5(firstH)
            ob6_t = self.g_output_unit_b6(firstH)
            ob7_t = self.g_output_unit_b7(firstH)


            g_predictions = g_predictions.write(i, o_t)
            g_predictions_dur = g_predictions_dur.write(i, tf.stack([ob1_t,ob2_t,ob3_t,ob4_t,ob5_t,ob6_t,ob7_t]))
            x_tp1 = ta_emb_x.read(i)
            a_tp1 = ta_emb_x_dur.read(i)

            next_a = self.x_dur[i]
            next_token = self.x[i]
            dur = duration_tensor_to_beat_duration(next_a)
            return i + 1, beat+dur,\
                self.chordKeys_onehot[(i+dur) % self.sequence_length],self.chordNotes[(i+dur) % self.sequence_length],self.lows[i],self.highs[i],\
                tf.constant(0,dtype=tf.int32), next_a,\
                tf.multiply(rep_count,tf.to_int32(tf.equal(prev_token,next_token)))+1, next_token,\
                x_tp1, a_tp1, firstH,\
                g_predictions, g_predictions_dur

        _, _, _, _, _, _, _, _, _, _, _, _, _, self.g_predictions, self.g_predictions_dur = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),duration_tensor_to_beat_duration(self.start_duration),
                self.chordKeys_onehot[0],self.chordNotes[0],tf.constant(0.0,dtype=tf.float32),tf.constant(0.0,dtype=tf.float32),
                tf.constant(0,dtype=tf.int32),self.start_duration,
                tf.constant(0,dtype=tf.int32),self.start_pitch,
                tf.gather(self.g_embeddings, self.start_pitch), self.start_duration,self.h0,
                g_predictions,g_predictions_dur))
        self.g_predictions = tf.reshape(
                self.g_predictions.stack(),
                [self.sequence_length, self.num_emb])
        self.g_predictions_dur = tf.reshape(
                self.g_predictions_dur.stack(),
                [self.sequence_length, 7, 2])

        # TODO JUMPPOINT
        # calculate discriminator loss
        self.d_gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.d_gen_predictions, labels=tf.zeros([self.sequence_length])))
        self.d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.d_real_predictions, labels=tf.ones([self.sequence_length])))

        # calculate generator rewards and loss
        decays = tf.exp(tf.log(self.reward_gamma) * tf.to_float(tf.range(self.sequence_length)))
        rewards = _backwards_cumsum(decays * tf.sigmoid(self.d_gen_predictions),
                                    self.sequence_length)
        #zero_pads = tf.zeros([self.max_sequence_length - self.sequence_length],tf.float32)
        r_div = tf.div(rewards, _backwards_cumsum(decays, self.sequence_length))
        expected_reward_short = tf.slice(self.expected_reward,[0],[self.sequence_length])
        normalized_rewards = r_div - expected_reward_short #\
            #tf.concat([r_div, zero_pads], 0) - self.expected_reward

        self.reward_loss = tf.reduce_mean(normalized_rewards ** 2)
        
        # TODO: ADD dur TO LOSS
        self.g_loss = \
            -tf.reduce_mean(tf.log(tf.clip_by_value(self.gen_o.stack(),1e-7,1e7)) * normalized_rewards) \
            -tf.reduce_mean(tf.log(tf.clip_by_value(self.gen_o_dur.stack(),1e-7,1e7)) * normalized_rewards)

        # pretraining loss
        self.pretrain_loss = \
            (-tf.reduce_sum(
                tf.one_hot(tf.to_int64(self.x),
                           self.num_emb, 1.0, 0.0) * tf.log(tf.clip_by_value(self.g_predictions,1e-7,1e7)))
             / tf.to_float(self.sequence_length)) \
            + (-tf.reduce_sum(
                tf.one_hot(tf.to_int64(self.x_dur),
                           2, 1.0, 0.0) * tf.log(tf.clip_by_value(self.g_predictions_dur,1e-7,1e7)))
             / tf.to_float(self.sequence_length))

        # training updates
        d_opt = self.d_optimizer(self.learning_rate)
        g_opt = self.g_optimizer(self.learning_rate)
        pretrain_opt = self.g_optimizer(self.learning_rate)
        reward_opt = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.d_gen_grad = tf.gradients(self.d_gen_loss, self.d_params)
        self.d_real_grad = tf.gradients(self.d_real_loss, self.d_params)
        self.d_gen_updates = d_opt.apply_gradients(zip(self.d_gen_grad, self.d_params))
        self.d_real_updates = d_opt.apply_gradients(zip(self.d_real_grad, self.d_params))

        self.reward_grad = tf.gradients(self.reward_loss, [self.expected_reward])
        self.reward_updates = reward_opt.apply_gradients(zip(self.reward_grad, [self.expected_reward]))

        self.g_grad = tf.gradients(self.g_loss, self.g_params)
        self.g_updates = g_opt.apply_gradients(zip(self.g_grad, self.g_params))

        self.pretrain_grad = tf.gradients(self.pretrain_loss, self.g_params)
        self.pretrain_updates = pretrain_opt.apply_gradients(zip(self.pretrain_grad, self.g_params))
        self.gen_x_tensor = tf.one_hot(self.gen_x_dur.stack(),7)

    def generate(self, session,chordkeys,chordkeys_onehot,chordnotes,sequence_length,start_pitch,start_duration):
        outputs = session.run(
                [self.gen_x, self.gen_x_tensor],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dim),
                           self.h0_dur: np.random.normal(size=self.hidden_dim_dur),
                           self.h0s: np.array([np.random.normal(size=self.hidden_dim) for _ in range(self.num_hidden_layers)]),
                           self.h0s_dur: np.array([np.random.normal(size=self.hidden_dim_dur) for _ in range(self.num_hidden_layers)]),
                           self.samples: np.random.random(sequence_length),
                           self.chordKeys:chordkeys, self.chordKeys_onehot:chordkeys_onehot, self.chordNotes: chordnotes,
                           self.sequence_length: sequence_length, self.start_pitch:start_pitch, self.start_duration:start_duration})
        return outputs

    def train_g_step(self, session,chordkeys,chordkeys_onehot,chordnotes,sequence_length,start_pitch,start_duration):
        outputs = session.run(
                [self.g_updates, self.reward_updates, self.g_loss,
                 self.expected_reward, self.gen_x, self.gen_x_tensor],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dim),
                           self.h0_dur: np.random.normal(size=self.hidden_dim_dur),
                           self.h0s: np.array([np.random.normal(size=self.hidden_dim) for _ in range(self.num_hidden_layers)]),
                           self.h0s_dur: np.array([np.random.normal(size=self.hidden_dim_dur) for _ in range(self.num_hidden_layers)]),
                           self.samples: np.random.random(sequence_length),
                           self.chordKeys:chordkeys, self.chordKeys_onehot:chordkeys_onehot, self.chordNotes: chordnotes,
                           self.sequence_length: sequence_length, self.start_pitch:start_pitch, self.start_duration:start_duration})
        return outputs

    def train_d_gen_step(self, session,chordkeys,chordkeys_onehot,chordnotes,sequence_length,start_pitch,start_duration):
        outputs = session.run(
                [self.d_gen_updates, self.d_gen_loss],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dim),
                           self.h0_dur: np.random.normal(size=self.hidden_dim_dur),
                           self.h0s: np.array([np.random.normal(size=self.hidden_dim) for _ in range(self.num_hidden_layers)]),
                           self.h0s_dur: np.array([np.random.normal(size=self.hidden_dim_dur) for _ in range(self.num_hidden_layers)]),
                           self.samples: np.random.random(sequence_length),
                           self.chordKeys:chordkeys, self.chordKeys_onehot:chordkeys_onehot, self.chordNotes: chordnotes,
                           self.sequence_length: sequence_length, self.start_pitch:start_pitch, self.start_duration:start_duration})
        return outputs

    def train_d_real_step(self, session, x, x_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,start_pitch,start_duration):
        outputs = session.run([self.d_real_updates, self.d_real_loss],
                              feed_dict={self.x: x, self.x_dur: x_dur,
                                         self.chordKeys:chordkeys, self.chordKeys_onehot:chordkeys_onehot,self.chordNotes:chordnotes, self.lows:low, self.highs:high,
                                         self.sequence_length: sequence_length, self.start_pitch:start_pitch, self.start_duration:start_duration})
        return outputs

    def pretrain_step(self, session, x, x_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,start_pitch,start_duration):
        outputs = session.run([self.pretrain_updates, self.pretrain_loss, self.g_predictions, self.g_predictions_dur],
                              feed_dict={self.x: x, self.x_dur: x_dur,
                                         self.h0_dur: np.random.normal(size=self.hidden_dim_dur),
                                         self.h0: np.random.normal(size=self.hidden_dim),
                                         self.h0s: np.array([np.random.normal(size=self.hidden_dim) for _ in range(self.num_hidden_layers)]),
                                         self.h0s_dur: np.array([np.random.normal(size=self.hidden_dim_dur) for _ in range(self.num_hidden_layers)]),
                                         self.chordKeys:chordkeys, self.chordKeys_onehot:chordkeys_onehot,self.chordNotes:chordnotes, self.lows:low, self.highs:high,
                                         self.sequence_length: sequence_length, self.start_pitch:start_pitch, self.start_duration:start_duration,
                                         self.temph0s: np.random.normal(size=self.hidden_dim), 
                                         self.temph0s_dur: np.random.normal(size=self.hidden_dim_dur)})
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1, dtype = tf.float32)

    def init_vector(self, shape):
        return tf.zeros(shape, dtype = tf.float32)

    # This method seems to be overridden by the GRU create_recurrent_unit() method.
    # Commented out for clarity.
    # 
    # def create_recurrent_unit(self, params):
    #     self.W_rec = tf.Variable(self.init_matrix([self.hidden_dim, self.emb_dim]))
    #     params.append(self.W_rec)
    #     def unit(x_t, h_tm1):
    #         return h_tm1 + tf.reshape(tf.matmul(self.W_rec, tf.reshape(x_t, [self.emb_dim, 1])), [self.hidden_dim])
    #     return unit

    def create_output_unit_no_emb(self,num_emb,hidden_dim,params):
        W_out = tf.Variable(self.init_matrix([num_emb,hidden_dim]))
        b_out = tf.Variable(self.init_vector([num_emb, 1]))
        params.extend([
            W_out,b_out])
        def unit(h_t):
            logits = tf.reshape(
                    b_out + tf.matmul(W_out, tf.reshape(h_t,[hidden_dim, 1])),
                    [1,num_emb])
            return tf.reshape(tf.nn.softmax(logits), [num_emb])
        return unit

    def create_output_unit(self, num_emb, emb_dim, hidden_dim,params, embeddings):
        W_out = tf.Variable(self.init_matrix([emb_dim, hidden_dim]))
        b_out1 = tf.Variable(self.init_vector([emb_dim, 1]))
        b_out2 = tf.Variable(self.init_vector([num_emb, 1]))
        params.extend([
            W_out, b_out1, b_out2])
        def unit(embeddings,num_emb,hidden_dim,h_t):
            logits = tf.reshape(
                    b_out2 +
                    tf.matmul(embeddings,
                              tf.tanh(b_out1 +
                                      tf.matmul(W_out, tf.reshape(h_t, [hidden_dim, 1])))),
                    [1, num_emb])
            return tf.reshape(tf.nn.softmax(logits), [num_emb])
        return unit

    def create_classifier_unit(self, params):
        W_class = tf.Variable(self.init_matrix([1, self.hidden_dim]))
        #W_class_dur = tf.Variable(self.init_matrix([1, self.hidden_dim_dur]))
        b_class = tf.Variable(self.init_vector([1]))
        params.extend([W_class, b_class])#, W_class_dur])
        def unit(h_t):
            return b_class + tf.matmul(W_class, tf.reshape(h_t, [self.hidden_dim, 1]))# \
                #+ tf.matmul(W_class_dur, tf.reshape(h_t_dur, [self.hidden_dim_dur,1]))
        return unit

    def d_optimizer(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)

    def g_optimizer(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)


class GRU(RNN):

    def create_recurrent_unit_dur(self, emb_dim,hidden_dim, params):
        W_rrepcount = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_zrepcount = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_hrepcount = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_rbeat = tf.Variable(self.init_matrix([hidden_dim, self.lenBeatVec]))
        W_zbeat = tf.Variable(self.init_matrix([hidden_dim, self.lenBeatVec]))
        W_hbeat = tf.Variable(self.init_matrix([hidden_dim, self.lenBeatVec]))
        W_rx = tf.Variable(self.init_matrix([hidden_dim, emb_dim]))
        W_zx = tf.Variable(self.init_matrix([hidden_dim, emb_dim]))
        W_hx = tf.Variable(self.init_matrix([hidden_dim, emb_dim]))
        U_rh = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        U_zh = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        U_hh = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        params.extend([
            W_rrepcount, W_zrepcount, W_hrepcount,
            W_rbeat, W_zbeat, W_hbeat,
            W_rx, W_zx, W_hx,
            U_rh, U_zh, U_hh])

        def unit(emb_dim,hidden_dim,x_t,beatVec, rep_count, h_tm1):

            rep_count = tf.reshape(tf.to_float(rep_count), [1,1])
            x_t = tf.reshape(x_t, [emb_dim, 1])
            beatVec = tf.reshape(beatVec, [self.lenBeatVec, 1])
            h_tm1 = tf.reshape(h_tm1, [hidden_dim, 1])
            r = tf.sigmoid(tf.matmul(W_rrepcount, rep_count) + \
                tf.matmul(W_rbeat, beatVec) + \
                tf.matmul(W_rx, x_t) + \
                tf.matmul(U_rh, h_tm1))
            z = tf.sigmoid(tf.matmul(W_zrepcount, rep_count) + \
                tf.matmul(W_zbeat, beatVec) + \
                tf.matmul(W_zx, x_t) + \
                tf.matmul(U_zh, h_tm1))
            h_tilda = tf.tanh(tf.matmul(W_hrepcount, rep_count) + \
                tf.matmul(W_hbeat, beatVec) + \
                tf.matmul(W_hx, x_t) + \
                tf.matmul(U_hh, r * h_tm1))
            h_t = (1 - z) * h_tm1 + z * h_tilda
            return tf.reshape(h_t, [hidden_dim])

        return unit

    def create_recurrent_unit_hidden(self, hidden_dim, params):
        W_rx = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        W_zx = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        W_hx = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        U_rh = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        U_zh = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        U_hh = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        params.extend([
            W_rx, W_zx, W_hx,
            U_rh, U_zh, U_hh])

        def unit(hidden_dim,x_t, h_tm1):

            x_t = tf.reshape(x_t, [hidden_dim, 1])
            h_tm1 = tf.reshape(h_tm1, [hidden_dim, 1])
            r = tf.sigmoid(
                tf.matmul(W_rx, x_t) + \
                tf.matmul(U_rh, h_tm1))
            z = tf.sigmoid(
                tf.matmul(W_zx, x_t) + \
                tf.matmul(U_zh, h_tm1))
            h_tilda = tf.tanh(
                tf.matmul(W_hx, x_t) + \
                tf.matmul(U_hh, r * h_tm1))
            h_t = (1 - z) * h_tm1 + z * h_tilda
            return tf.reshape(h_t, [hidden_dim])

        return unit

    def create_recurrent_unit_pitch(self, emb_dim,emb_dim_dur,hidden_dim, params):
        W_rlow = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_zlow = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_hlow = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_rhigh = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_zhigh = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_hhigh = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_rcnote = tf.Variable(self.init_matrix([hidden_dim, 12]))
        W_zcnote = tf.Variable(self.init_matrix([hidden_dim, 12]))
        W_hcnote = tf.Variable(self.init_matrix([hidden_dim, 12]))
        W_rckey = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_zckey = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_hckey = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_rrepcount = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_zrepcount = tf.Variable(self.init_matrix([hidden_dim, 1]))
        W_hrepcount = tf.Variable(self.init_matrix([hidden_dim, 1]))
        #W_racount = tf.Variable(self.init_matrix([hidden_dim, 1]))
        #W_zacount = tf.Variable(self.init_matrix([hidden_dim, 1]))
        #W_hacount = tf.Variable(self.init_matrix([hidden_dim, 1]))
        #W_rbeat = tf.Variable(self.init_matrix([hidden_dim, self.lenBeatVec]))
        #W_zbeat = tf.Variable(self.init_matrix([hidden_dim, self.lenBeatVec]))
        #W_hbeat = tf.Variable(self.init_matrix([hidden_dim, self.lenBeatVec]))
        W_ra = tf.Variable(self.init_matrix([hidden_dim, 7]))
        W_za = tf.Variable(self.init_matrix([hidden_dim, 7]))
        W_ha = tf.Variable(self.init_matrix([hidden_dim, 7]))
        W_rx = tf.Variable(self.init_matrix([hidden_dim, emb_dim]))
        W_zx = tf.Variable(self.init_matrix([hidden_dim, emb_dim]))
        W_hx = tf.Variable(self.init_matrix([hidden_dim, emb_dim]))
        U_rh = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        U_zh = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        U_hh = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        params.extend([
            W_rlow, W_zlow, W_hlow,
            W_rhigh, W_zhigh, W_hhigh,
            W_rcnote, W_zcnote, W_hcnote,
            W_rckey, W_zckey, W_hckey,
            W_rrepcount, W_zrepcount, W_hrepcount,
            #W_racount, W_zacount, W_hacount,
            #W_rbeat, W_zbeat, W_hbeat,
            W_rx, W_zx, W_hx,
            W_ra, W_za, W_ha,
            U_rh, U_zh, U_hh])

        def unit(emb_dim,hidden_dim,x_t,a_t,beatVec, rep_count, h_tm1,chordkey_vec,chordnote_vec,low,high):

            high = tf.reshape(tf.to_float(high), [1,1])
            low = tf.reshape(tf.to_float(low), [1,1])
            chordnote_vec = tf.reshape(tf.to_float(chordnote_vec), [12,1])
            chordkey_vec = tf.reshape(tf.to_float(chordkey_vec), [1,1])
            rep_count = tf.reshape(tf.to_float(rep_count), [1,1])
            #a_count = tf.reshape(tf.to_float(a_count), [1,1])
            x_t = tf.reshape(tf.to_float(x_t), [emb_dim, 1])
            a_t = tf.reshape(tf.to_float(a_t), [7, 1])
            #beatVec = tf.reshape(beatVec, [self.lenBeatVec, 1])
            h_tm1 = tf.reshape(tf.to_float(h_tm1), [hidden_dim, 1])
            r = tf.sigmoid(tf.matmul(W_rrepcount, rep_count) + \
                #tf.matmul(W_racount, a_count) + \
                tf.matmul(W_rckey,chordkey_vec) + \
                tf.matmul(W_rcnote, chordnote_vec) + \
                tf.matmul(W_rlow, low) + \
                tf.matmul(W_rhigh, high) + \
                #tf.matmul(W_rbeat, beatVec) + \
                tf.matmul(W_rx, x_t) + \
                tf.matmul(tf.to_float(W_ra), tf.to_float(a_t)) + \
                tf.matmul(U_rh, h_tm1))
            z = tf.sigmoid(tf.matmul(W_zrepcount, rep_count) + \
                #tf.matmul(W_zacount, a_count) + \
                tf.matmul(W_zckey,chordkey_vec) + \
                tf.matmul(W_zcnote, chordnote_vec) + \
                tf.matmul(W_zlow, low) + \
                tf.matmul(W_zhigh, high) + \
                #tf.matmul(W_zbeat, beatVec) + \
                tf.matmul(W_zx, x_t) + \
                tf.matmul(W_za, a_t) + \
                tf.matmul(U_zh, h_tm1))
            h_tilda = tf.tanh(tf.matmul(W_hrepcount, rep_count) + \
                #tf.matmul(W_hacount, a_count) + \
                tf.matmul(W_hckey,chordkey_vec) + \
                tf.matmul(W_hcnote, chordnote_vec) + \
                tf.matmul(W_hlow, low) + \
                tf.matmul(W_hhigh, high) + \
                #tf.matmul(W_hbeat, beatVec) + \
                tf.matmul(W_hx, x_t) + \
                tf.matmul(W_ha, a_t) + \
                tf.matmul(U_hh, r * h_tm1))
            h_t = (1 - z) * h_tm1 + z * h_tilda
            return tf.reshape(h_t, [hidden_dim])

        return unit
