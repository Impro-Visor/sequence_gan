__doc__ = """RNN-based GAN.  For applying Generative Adversarial Networks to sequential data."""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops

ONE_HOT = 0
BITS = 1

class RNN(object):

    def __init__(self, num_emb, num_emb_dur, num_emb_dura, num_emb_chord,
                 emb_dim, emb_dim_dur, emb_dim_dura, emb_dim_chord,
                 hidden_dim,hidden_dim_b, num_hidden_layers,
                 max_sequence_length, max_block_length, endCount,
                 learning_rate=0.01, reward_gamma=0.9,MIDI_MIN=55,MIDI_MAX=89,ENCODING=ONE_HOT):
        self.num_emb = num_emb              # Pitch categories
        self.num_emb_dur = num_emb_dur      # Duration categories
        self.num_emb_dura = num_emb_dura    # Alt duration categories
        self.num_emb_chord = num_emb_chord  # Chord key categories
        self.emb_dim = emb_dim              # Embedding size for pitches
        self.emb_dim_dur = emb_dim_dur      # Embedding size for durations
        self.emb_dim_dura = emb_dim_dura    # Embedding size for alt durations
        self.emb_dim_chord = emb_dim_chord  # Embedding size for chord keys
        self.hidden_dim = hidden_dim        # Hidden layer size 
        self.hidden_dim_b = hidden_dim_b    # Hidden layer size for blocks

        self.emb_dim_d = int(emb_dim/8)             # Discriminator embedding size for pitches
        self.emb_dim_dur_d = int(emb_dim_dur/8)     # Discriminator embedding size for durations
        self.emb_dim_dura_d = int(emb_dim_dura/8)   # Discriminator embedding size for alt durations
        self.emb_dim_chord_d = int(emb_dim_chord/8) # Discriminator embedding size for chord keys
        self.hidden_dim_d = int(hidden_dim/10)      # Hidden layer size for discriminator
        self.hidden_dim_b_d = int(hidden_dim_b/10)  # Hidden layer size for discriminator blocsk

        self.num_hidden_layers = num_hidden_layers      # Number of hidden layers
        self.max_sequence_length = max_sequence_length  # Max sequence length over corpus
        self.maxblocklength = max_block_length          # Max block length
        self.endCount = endCount                        # Countdown till end of sequence
        self.sequence_length = tf.placeholder(tf.int32,shape=[],name="sequence_length") # Current sequence_length
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)         # Learning rate
        self.reward_gamma = reward_gamma                                                
        self.g_params = [] # List of parameters to train for the generator
        self.d_params = [] # List of parameters to train for the discriminator
        self.MIDI_MIN = MIDI_MIN # lowest note value found in trainingset
        self.MIDI_MAX = MIDI_MAX # highest note value found in trainingset
        self.REST_VAL = self.MIDI_MAX-self.MIDI_MIN+1

        self.expected_reward = tf.Variable(tf.zeros([self.max_sequence_length]))

        self.num_features_hidden = (0 #+
                                    + self.hidden_dim # x
                                    + self.hidden_dim # h
                                    + self.hidden_dim # c, peephole
                                    )
        self.num_inputs_hidden = self.num_features_hidden - self.hidden_dim - self.hidden_dim # - h - c

        self.num_features_miniseq = (0 +
                                    + self.maxblocklength*self.emb_dim_dura # dura block
                                    + self.maxblocklength*self.emb_dim_dur # dur block
                                    + self.maxblocklength*self.emb_dim # note block
                                    + self.hidden_dim_b # h
                                    + self.hidden_dim_b # c, peephole
                                    )
        self.num_inputs_miniseq = self.num_features_miniseq - self.hidden_dim_b - self.hidden_dim_b # - h - c

        self.num_features_block = (0 #+
                                    #+ 1 # low
                                    #+ 1 # high
                                    + 12 # chord notes
                                    + self.emb_dim_chord # chord key
                                    + 1 # acount
                                    + 1 # duracount
                                    + 1 # repcount
                                    + 1 # prev interval
                                    + 12 # prev chord pitch
                                    #+ self.hidden_dim_b # block
                                    #+ 48 # beat
                                    + self.endCount # countdown till end of sequence
                                    + self.emb_dim_dur # a0
                                    + self.emb_dim_dur # a1
                                    + self.emb_dim_dur # a2
                                    + self.emb_dim_dur # a3
                                    + self.emb_dim_dura # dura
                                    + self.emb_dim # x0
                                    + self.emb_dim # x1
                                    + self.emb_dim # x2
                                    + self.emb_dim # x3
                                    + self.hidden_dim # h
                                    + self.hidden_dim # c, peephole
                                    )
        self.num_features_noblock_d = (0 #+
                                    #+ 1 # low
                                    #+ 1 # high
                                    + 12 # chord notes
                                    + self.emb_dim_chord_d # chord key
                                    + 1 # acount
                                    + 1 # duracount
                                    + 1 # repcount
                                    + 1 # prev interval
                                    + 12 # prev chord pitch
                                    #+ self.hidden_dim_b # block
                                    #+ 48 # beat
                                    + self.endCount # countdown till end of sequence
                                    + self.emb_dim_dur_d # a0
                                    + self.emb_dim_dur_d # a1
                                    + self.emb_dim_dur_d # a2
                                    + self.emb_dim_dur_d # a3
                                    + self.emb_dim_dura_d # dura
                                    + self.emb_dim_d # x0
                                    + self.emb_dim_d # x1
                                    + self.emb_dim_d # x2
                                    + self.emb_dim_d # x3
                                    + self.hidden_dim_d # h
                                    + self.hidden_dim_d # c, peephole
                                    )#-self.hidden_dim_b
        self.num_inputs_block = self.num_features_block - self.hidden_dim - self.hidden_dim # - h - c

        self.num_features_noblock = self.num_features_block# - self.hidden_dim_b
        self.num_inputs_noblock = self.num_features_noblock - self.hidden_dim - self.hidden_dim # - h - c
        self.num_inputs_noblock_d = self.num_features_noblock_d - self.hidden_dim_d - self.hidden_dim_d

        with tf.variable_scope('generator'):
            # Embedding matrix for notes
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)

            # Embedding matrix for dur bit
            self.g_embeddings_dur = tf.Variable(self.init_matrix([self.num_emb_dur,self.emb_dim_dur]))
            self.g_params.append(self.g_embeddings_dur)

            # Embedding matrix for duration lengths
            self.g_embeddings_dura = tf.Variable(self.init_matrix([self.num_emb_dura,self.emb_dim_dura]))
            self.g_params.append(self.g_embeddings_dur)

            # Embedding matrix for chord keys
            self.g_embeddings_chord = tf.Variable(self.init_matrix([self.num_emb_chord,self.emb_dim_chord]))
            self.g_params.append(self.g_embeddings_chord)

            # Recurrent units mapping input to h_t
            self.g_recurrent_unit_miniseq = self.create_lstm_layer(self.hidden_dim_b, self.num_features_miniseq, self.g_params)
            self.g_recurrent_unit = self.create_lstm_layer(self.hidden_dim, self.num_features_block, self.g_params)
            self.g_hidden = self.create_lstm_layer(self.hidden_dim, self.num_features_hidden, self.g_params)

            # Output units mapping h_t to token logits
            self.g_output_unit = self.create_output_unit(self.num_emb, self.emb_dim, self.hidden_dim, self.g_params, self.g_embeddings)
            self.g_output_unit_dur = self.create_output_unit(self.num_emb_dur, self.emb_dim_dur, self.hidden_dim, self.g_params, self.g_embeddings_dur)

        with tf.variable_scope('discriminator'):
            # Embedding matrix for notes
            self.d_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim_d]))
            self.d_params.append(self.d_embeddings)

            # Embedding matrix for dur bit
            self.d_embeddings_dur = tf.Variable(self.init_matrix([self.num_emb_dur, self.emb_dim_dur_d]))
            self.d_params.append(self.d_embeddings_dur)

            # Embedding matrix for duration lengths
            self.d_embeddings_dura = tf.Variable(self.init_matrix([self.num_emb_dura, self.emb_dim_dura_d]))
            self.d_params.append(self.d_embeddings_dura)

            # Embedding matrix for chord keys
            self.d_embeddings_chord = tf.Variable(self.init_matrix([self.num_emb_chord,self.emb_dim_chord_d]))
            self.d_params.append(self.d_embeddings_chord)

            self.d_recurrent_unit = self.create_lstm_layer(self.hidden_dim_d, self.num_features_noblock_d, self.d_params)

            # Output unit mapping h_t to class prediction logits
            self.d_classifier_unit = self.create_classifier_unit_d(self.d_params)

            # Initial horizontal inputs for discriminator recurrent units
            self.d_h0 = tf.Variable(self.init_vector([self.hidden_dim_d]))
            self.d_c0 = tf.Variable(self.init_vector([self.hidden_dim_d]))
            self.d_params.append(self.d_h0)
            self.d_params.append(self.d_c0)

        self.h0 = tf.random_normal([self.hidden_dim]) # initial horizontal vector
        self.c0 = tf.random_normal([self.hidden_dim]) # initial state vector
        self.blockh0 = tf.random_normal([self.hidden_dim_b]) # init h vec for blocks
        self.blockc0 = tf.random_normal([self.hidden_dim_b]) # init state vec for blocks
        self.lengths = tf.placeholder(tf.int32, shape=[None],name="lengths") # sequence of lengths for minisequences
        self.lengths_length = tf.shape(self.lengths)[0] # length of sequence of lengths
        self.x = tf.placeholder(tf.int32, shape=[None],name="x")  # sequence of indices of true note intervals, not including start token
        self.x_dur = tf.placeholder(tf.int32, shape=[None],name="x_dur")  # sequence of indices of true durs, not including start token
        self.x_pitch = tf.placeholder(tf.int32,shape=[None],name="x_pitch") # sequence of indices of true notes, not including start token
        self.samples = tf.placeholder(tf.float32,shape=[None],name="samples")#tf.random_uniform([self.sequence_length]) # random samples from [0, 1]
        self.samples_dur = tf.placeholder(tf.float32,shape=[None],name="samples_dur")#tf.random_uniform([self.sequence_length])  # random samples from [0, 1]
        self.chordKeys = tf.placeholder(tf.int32, shape=[None],name="chordKeys") # sequence of chord key values
        self.chord_length = tf.shape(self.chordKeys)[0] # Get length of sequence of chord key values, i.e. how long the sequence should go for
        #self.start_pitch = tf.placeholder(tf.int32,shape=[]) # starting pitch for INTERVALs
        self.p0 = tf.placeholder(tf.int32,shape=[],name="p0") # starting pitch
        self.d0 = tf.placeholder(tf.int32,shape=[],name="d0") # starting duration
        self.p1 = tf.placeholder(tf.int32,shape=[],name="p1") # starting pitch
        self.d1 = tf.placeholder(tf.int32,shape=[],name="d1") # starting duration
        self.p2 = tf.placeholder(tf.int32,shape=[],name="p2") # starting pitch
        self.d2 = tf.placeholder(tf.int32,shape=[],name="d2") # starting duration
        self.p3 = tf.placeholder(tf.int32,shape=[],name="p3") # starting pitch
        self.d3 = tf.placeholder(tf.int32,shape=[],name="d3") # starting duration
        #self.start_duration = tf.placeholder(tf.int32,shape=[]) # starting duration
        self.start_dura = tf.placeholder(tf.int32, shape=[],name="start_dura") # starting duration length
        self.start_beat = tf.constant(0,dtype=tf.int32)#tf.placeholder(tf.int32,shape=[]) # starting beat count
        self.start_chordkey = tf.placeholder(tf.int32,shape=[],name="start_chordkey") # starting chord key
        self.chordKeys_onehot = tf.placeholder(tf.int32, shape=[None],name="chordKeys_onehot") # sequence of chord keys, onehot
        self.chordNotes = tf.placeholder(tf.int32, shape=[None, 12],name="chordNotes") # sequence of vectors of notes in the chord
        self.lows = tf.placeholder(tf.float32, shape=[None],name="lows") # sequence of low pos ratios
        self.highs = tf.placeholder(tf.float32, shape=[None],name="highs") # sequence of high pos ratios

        ############################################################################################################

        # generator on initial randomness
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.chord_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.chord_length,
                                             dynamic_size=False, infer_shape=True)
        gen_o_dur = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.chord_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x_dur = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.chord_length,
                                             dynamic_size=False, infer_shape=True)

        samples = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.chord_length)
        samples = samples.unstack(self.samples)

        samples_dur = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.chord_length)
        samples_dur = samples_dur.unstack(self.samples_dur)

        # RNN loop filling tensorarrays up to the chord seq length

        def _newg_recurrence(ii,seq_count, beat_count, pitch_count, prev_interval, prev_pitch_chord,
            chordkey_vec, chordnote_vec,
            a_count, prev_a, 
            dura_count, prev_dura,
            rep_count, prev_token,
            x_t,a_t, dura_t, h_tm1, c_tm1,
            x1,x2,x3,a1,a2,a3,
            gen_o, gen_x, gen_o_dur, gen_x_dur): 

            timeTillEnd = self.chord_length - ii -1 -1 # -1 to boost ii, -1 for onehot indexing
            timeTillEndVec = tf.reshape(tf.one_hot(timeTillEnd,self.endCount,on_value=1.0,off_value=0.0), [self.endCount])
            
            # Feed pitch inputs to input GRU layer of pitch RNN
            rchordnote_vec = tf.reshape(tf.to_float(chordnote_vec), [12])
            rchordkey_vec = tf.reshape(tf.gather(self.g_embeddings_chord,chordkey_vec), [self.emb_dim_chord])
            rrep_count = tf.reshape(tf.to_float(rep_count), [1])
            ra_count = tf.reshape(tf.to_float(a_count), [1])
            rdura_count = tf.reshape(tf.to_float(dura_count),[1])
            rprev_interval = tf.reshape(tf.to_float(prev_interval),[1])
            rprev_pitch_chord = tf.reshape(tf.one_hot(prev_pitch_chord,12,on_value=1.0,off_value=0.0),[12])
            #rblock_input = tf.reshape(tf.to_float(block_input), [self.hidden_dim_b])
            rx_t = tf.reshape(x_t, [self.emb_dim])
            ra_t = tf.reshape(a_t, [self.emb_dim_dur])
            rx1_t = tf.reshape(tf.gather(self.g_embeddings, x1), [self.emb_dim])
            ra1_t = tf.reshape(tf.gather(self.g_embeddings_dur, a1), [self.emb_dim_dur])
            rx2_t = tf.reshape(tf.gather(self.g_embeddings, x2), [self.emb_dim])
            ra2_t = tf.reshape(tf.gather(self.g_embeddings_dur, a2), [self.emb_dim_dur])
            rx3_t = tf.reshape(tf.gather(self.g_embeddings, x3), [self.emb_dim])
            ra3_t = tf.reshape(tf.gather(self.g_embeddings_dur, a3), [self.emb_dim_dur])
            rdura_t = tf.reshape(dura_t,[self.emb_dim_dura])
            #rbeatVec = tf.reshape(beatVec, [48])

            rc_tm1 = tf.reshape(c_tm1, [self.hidden_dim, 1])
            rh_tm1 = tf.reshape(h_tm1, [self.hidden_dim, 1])
            inputs_recurr = tf.reshape(
                tf.concat([rchordnote_vec,rchordkey_vec,rrep_count,ra_count,rdura_count,rprev_interval,rprev_pitch_chord,timeTillEndVec,rx_t,rx1_t,rx2_t,rx3_t,ra_t,ra1_t,ra2_t,ra3_t,rdura_t],
                #tf.concat([rhigh,rlow,rchordnote_vec,rchordkey_vec,rrep_count,ra_count,rdura_count,rprev_interval,rprev_pitch_chord,rblock_input,rx_t,ra_t,rdura_t,rbeatVec],
                    0), 
                [self.num_inputs_block,1])

            firstH,firstC = self.g_recurrent_unit(rh_tm1,inputs_recurr,rc_tm1)
            firstH = tf.reshape(firstH, [self.hidden_dim])
            firstC = tf.reshape(firstC, [self.hidden_dim])

            # Feed output to softmax unit to get next predicted token
            o_t = self.g_output_unit(self.g_embeddings, self.num_emb, self.hidden_dim, firstH)
            o_cumsum = tf.cumsum(o_t)  # prepare for sampling
            sample = samples.read(seq_count)
            sample_dur = samples_dur.read(seq_count)
            next_token = tf.maximum(tf.to_int32(tf.reduce_min(tf.where(sample < o_cumsum))),tf.constant(0,dtype=tf.int32))   # sample
            x_tp1 = tf.gather(self.g_embeddings, next_token)

            oa_t = self.g_output_unit_dur(self.g_embeddings_dur, self.num_emb_dur, self.hidden_dim, firstH)
            oa_cumsum = tf.cumsum(oa_t)
            next_token_dur = tf.maximum(tf.to_int32(tf.reduce_min(tf.where(sample_dur < oa_cumsum))),tf.constant(0,dtype=tf.int32))
            a_tp1 = tf.gather(self.g_embeddings_dur, next_token_dur)
            next_token_dura = tf.mod(next_token_dur + 48 - beat_count-1,tf.constant(48,dtype=tf.int32))
            dura_tp1 = tf.gather(self.g_embeddings_dura, next_token_dura)

            gen_o = gen_o.write(seq_count, tf.gather(o_t, next_token))  # we only need the sampled token's probability
            gen_x = gen_x.write(seq_count, next_token)  # indices, not embeddings

            gen_o_dur = gen_o_dur.write(seq_count, tf.gather(oa_t, next_token_dur))  # we only need the sampled token's probability
            gen_x_dur = gen_x_dur.write(seq_count, next_token_dur)  # indices, not embeddings

            newPitch = tf.cond(tf.equal(next_token,tf.constant(self.REST_VAL,dtype=tf.int32)), lambda: pitch_count, lambda: next_token )
            cpitch = tf.cond(tf.equal(next_token,tf.constant(self.REST_VAL,dtype=tf.int32)), lambda: tf.constant(0,dtype=tf.int32), lambda: tf.mod(12+next_token-chordkey_vec,12))
            newInterval = newPitch - pitch_count
            newii = ii + next_token_dura+1 # BEAT POSITION / DURATION ENCODING
            #newii = ii + 1 # TIMESTEP ENCODING

            return newii,seq_count+1, next_token_dur, newPitch, newInterval, cpitch,\
                self.chordKeys_onehot[tf.minimum(newii,self.chord_length-1)],self.chordNotes[tf.minimum(newii,self.chord_length-1)],\
                tf.multiply(a_count,tf.to_int32(tf.equal(prev_a,next_token_dur)))+1,next_token_dur, \
                tf.multiply(a_count,tf.to_int32(tf.equal(prev_dura,next_token_dura)))+1,next_token_dura, \
                tf.multiply(rep_count,tf.to_int32(tf.equal(prev_token,next_token)))+1,next_token, \
                x_tp1, a_tp1, dura_tp1, firstH,firstC,\
                next_token,x1,x2, next_token_dur,a1,a2,\
                gen_o, gen_x,gen_o_dur,gen_x_dur

        ii,self.gen_sequence_length, beat_count, pitch_count, prev_interval, prev_pitch_chord,\
        chordkey_vec, chordnote_vec,\
        a_count, prev_a,\
        dura_count, prev_dura,\
        rep_count, prev_token,\
        x_t,a_t,dura_t, h_tm1, c_tm1,\
        x1,x2,x3,a1,a2,a3,\
        self.gen_o, self.gen_x, self.gen_o_dur, self.gen_x_dur= control_flow_ops.while_loop(
            cond = lambda \
                ii,seq_count, beat_count, pitch_count, prev_interval, prev_pitch_chord,\
                chordkey_vec, chordnote_vec,\
                a_count, prev_a,\
                dura_count, prev_dura,\
                rep_count, prev_token,\
                x_t,a_t,dura_t, h_tm1, c_tm1,\
                x1,x2,x3,a1,a2,a3,\
                gen_o, gen_x, gen_o_dur, gen_x_dur : ii < self.chord_length,
            body = _newg_recurrence,
            loop_vars = (
                tf.constant(0, dtype=tf.int32),tf.constant(0, dtype=tf.int32),self.d0,self.p0,tf.constant(0,dtype=tf.int32),tf.mod(12+self.p0-self.start_chordkey,12),
                self.chordKeys_onehot[0],self.chordNotes[0],
                tf.constant(1,dtype=tf.int32),self.d0,
                tf.constant(1,dtype=tf.int32),self.start_dura,
                tf.constant(1,dtype=tf.int32),self.p0,
                tf.gather(self.g_embeddings, self.p0),tf.gather(self.g_embeddings_dur, self.d0),tf.gather(self.g_embeddings_dura,self.start_dura),self.h0,self.c0,
                self.p1,self.p2,self.p3,self.d1,self.d2,self.d3,
                gen_o, gen_x,gen_o_dur,gen_x_dur
                )
            )

        def _newg_filler(ii,gen_o,gen_x,gen_o_dur,gen_x_dur):
            gen_o = gen_o.write(ii, 0.0)  # we only need the sampled token's probability
            gen_x = gen_x.write(ii, 0)  # indices, not embeddings
            gen_o_dur = gen_o_dur.write(ii, 0.0)  # we only need the sampled token's probability
            gen_x_dur = gen_x_dur.write(ii, 0)  # indices, not embeddings
            return ii+1,gen_o,gen_x,gen_o_dur,gen_x_dur

        # RNN loop filling tensorarrays up to the max
        ii,self.gen_o, self.gen_x, self.gen_o_dur, self.gen_x_dur = control_flow_ops.while_loop(
            cond = lambda ii,gen_o,gen_x,gen_o_dur,gen_x_dur : ii<self.chord_length,
            body = _newg_filler,
            loop_vars = (
                    self.gen_sequence_length,self.gen_o,self.gen_x,self.gen_o_dur,self.gen_x_dur
                )
            )
        self.gen_o = tf.slice(self.gen_o.stack(),[0],[self.gen_sequence_length])
        self.gen_x = tf.slice(self.gen_x.stack(),[0],[self.gen_sequence_length])
        self.gen_o_dur = tf.slice(self.gen_o_dur.stack(),[0],[self.gen_sequence_length])
        self.gen_x_dur = tf.slice(self.gen_x_dur.stack(),[0],[self.gen_sequence_length])

        #####################################################################################################################

        # discriminator on generated and real data: note vars
        d_gen_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size= self.gen_sequence_length,
            dynamic_size=False, infer_shape=True)
        d_real_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        self.gen_x_out = tf.identity(self.gen_x,name="gen_x_out")
        emb_gen_x = tf.gather(self.d_embeddings, self.gen_x)
        ta_emb_gen_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_gen_x = ta_emb_gen_x.unstack(emb_gen_x)

        emb_real_x = tf.gather(self.d_embeddings, self.x)
        ta_emb_real_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_real_x = ta_emb_real_x.unstack(emb_real_x)

        # discriminator on generated and real data: dur vars
        self.gen_x_dur_out = tf.identity(self.gen_x_dur,name="gen_x_dur_out")
        emb_gen_x_dur = tf.gather(self.d_embeddings_dur, self.gen_x_dur)
        ta_emb_gen_x_dur = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.gen_sequence_length)
        ta_emb_gen_x_dur = ta_emb_gen_x_dur.unstack(emb_gen_x_dur)

        emb_real_x_dur = tf.gather(self.d_embeddings_dur, self.x_dur)
        ta_emb_real_x_dur = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.gen_sequence_length)
        ta_emb_real_x_dur = ta_emb_real_x_dur.unstack(emb_real_x_dur)

        # discriminator recurrence
        def _d_recurrence(i,seqlen,chordindex,beat_count,prev_interval, prev_pitch_chord,
            chordkey_vec, chordnote_vec,
            a_count, prev_a, durs,
            dura_count, prev_dura,
            rep_count, prev_token, notes, 
            inputs, inputs_dur, h_tm1, c_tm1,
            x1,x2,x3,a1,a2,a3,
            pred):
            #beatVec = tf.one_hot(beat_count,48,1.0,0.0)
            notesTillEnd = seqlen - i -1 -1 # -1 to boost seqindex, -1 for onehot indexing
            notesTillEndVec = tf.reshape(tf.one_hot(notesTillEnd,self.endCount,on_value=1.0,off_value=0.0), [self.endCount])
            x_t = tf.gather(self.d_embeddings, prev_token)#inputs.read(i)
            next_token = notes[i]
            a_t = tf.gather(self.d_embeddings_dur, prev_a)#inputs_dur.read(i)
            next_a = durs[i]
            dura_t = tf.gather(self.d_embeddings_dura,prev_dura)
            next_token_dura = tf.mod(next_a+48-beat_count-1,tf.constant(48,dtype=tf.int32))

            rchordnote_vec = tf.reshape(tf.to_float(chordnote_vec), [12])
            rchordkey_vec = tf.reshape(tf.gather(self.d_embeddings_chord,chordkey_vec), [self.emb_dim_chord_d])
            rrep_count = tf.reshape(tf.to_float(rep_count), [1])
            ra_count = tf.reshape(tf.to_float(a_count), [1])
            rdura_count = tf.reshape(tf.to_float(dura_count),[1])
            rprev_interval = tf.reshape(tf.to_float(prev_interval),[1])
            rprev_pitch_chord = tf.reshape(tf.one_hot(prev_pitch_chord,12,on_value=1.0,off_value=0.0),[12])
            rx_t = tf.reshape(x_t, [self.emb_dim_d])
            ra_t = tf.reshape(a_t, [self.emb_dim_dur_d])
            rx1_t = tf.reshape(tf.gather(self.d_embeddings, x1), [self.emb_dim_d])
            ra1_t = tf.reshape(tf.gather(self.d_embeddings_dur, a1), [self.emb_dim_dur_d])
            rx2_t = tf.reshape(tf.gather(self.d_embeddings, x2), [self.emb_dim_d])
            ra2_t = tf.reshape(tf.gather(self.d_embeddings_dur, a2), [self.emb_dim_dur_d])
            rx3_t = tf.reshape(tf.gather(self.d_embeddings, x3), [self.emb_dim_d])
            ra3_t = tf.reshape(tf.gather(self.d_embeddings_dur, a3), [self.emb_dim_dur_d])
            rdura_t = tf.reshape(dura_t,[self.emb_dim_dura_d])
            #rbeatVec = tf.reshape(beatVec, [48])

            rc_tm1 = tf.reshape(c_tm1, [self.hidden_dim_d, 1])
            rh_tm1 = tf.reshape(h_tm1, [self.hidden_dim_d, 1])
            inputs_recurr = tf.reshape(
                tf.concat([rchordnote_vec,rchordkey_vec,rrep_count,ra_count,rdura_count,rprev_interval,rprev_pitch_chord,notesTillEndVec,rx_t,rx1_t,rx2_t,rx3_t,ra_t,ra1_t,ra2_t,ra3_t,rdura_t],
                #tf.concat([rhigh,rlow,rchordnote_vec,rchordkey_vec,rrep_count,ra_count,rdura_count,rprev_interval,rprev_pitch_chord,rx_t,ra_t,rdura_t,rbeatVec],
                    0), 
                [self.num_inputs_noblock_d,1])

            h_t,c_t = self.d_recurrent_unit(rh_tm1,inputs_recurr,rc_tm1)
            h_t = tf.reshape(h_t, [self.hidden_dim_d])
            c_t = tf.reshape(c_t, [self.hidden_dim_d])

            y_t = self.d_classifier_unit(h_t)
            pred = pred.write(i, y_t)
            cpitch = tf.cond(tf.equal(next_token,tf.constant(self.REST_VAL,dtype=tf.int32)), lambda: tf.constant(0,dtype=tf.int32), lambda: tf.mod(12+next_token-chordkey_vec,12))
            newchordindex = tf.minimum(chordindex + next_token_dura+1,self.chord_length-1)


            return i + 1,seqlen,newchordindex, next_a,next_token-prev_token,cpitch,\
                self.chordKeys_onehot[newchordindex],self.chordNotes[newchordindex],\
                tf.multiply(a_count,tf.to_int32(tf.equal(prev_a,next_a)))+1, next_a,durs,\
                tf.multiply(dura_count,tf.to_int32(tf.equal(prev_dura,next_token_dura)))+1,next_token_dura, \
                tf.multiply(rep_count,tf.to_int32(tf.equal(prev_token,next_token)))+1, next_token,notes,\
                inputs, inputs_dur, h_t, c_t,\
                next_token,x1,x2, next_a,a1,a2,\
                pred

        i,seqlen,chordindex,beat_count,prev_interval, prev_pitch_chord,\
        chordkey_vec, chordnote_vec,\
        a_count, prev_a, durs,\
        dura_count, prev_dura,\
        rep_count, prev_token, notes,\
        inputs, inputs_dur, h_tm1, c_tm1,\
        x1,x2,x3,a1,a2,a3,\
        self.d_gen_predictions = control_flow_ops.while_loop(
            cond=lambda \
                i,seqlen,chordindex,beat_count,prev_interval, prev_pitch_chord,\
                chordkey_vec, chordnote_vec,\
                a_count, prev_a, durs,\
                dura_count, prev_dura,\
                rep_count, prev_token, notes,\
                inputs, inputs_dur, h_tm1, c_tm1,\
                x1,x2,x3,a1,a2,a3,\
                pred: i < self.gen_sequence_length,
            body=_d_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),self.gen_sequence_length,tf.constant(0,dtype=tf.int32), self.d0, tf.constant(0,dtype=tf.int32),tf.mod(12+self.p0-self.start_chordkey,12),
                self.chordKeys_onehot[0],self.chordNotes[0],
                tf.constant(1,dtype=tf.int32), self.d0, self.gen_x_dur,
                tf.constant(1,dtype=tf.int32), self.start_dura,
                tf.constant(1,dtype=tf.int32), self.p0, self.gen_x,
                ta_emb_gen_x, ta_emb_gen_x_dur,self.d_h0,self.d_c0, 
                self.p1,self.p2,self.p3,self.d1,self.d2,self.d3,
                d_gen_predictions))
        self.d_gen_predictions = tf.reshape(
                self.d_gen_predictions.stack(),
                [self.gen_sequence_length])
        self.d_gen_predictions_out = tf.identity(self.d_gen_predictions,name="d_gen_predictions_out")

        i,seqlen,chordindex,beat_count,prev_interval, prev_pitch_chord,\
        chordkey_vec, chordnote_vec,\
        a_count, prev_a, durs,\
        dura_count, prev_dura,\
        rep_count, prev_token, notes,\
        inputs, inputs_dur, h_tm1, c_tm1,\
        x1,x2,x3,a1,a2,a3,\
        self.d_real_predictions = control_flow_ops.while_loop(
            cond=lambda \
                i,seqlen,chordindex,beat_count,prev_interval, prev_pitch_chord,\
                chordkey_vec, chordnote_vec,\
                a_count, prev_a, durs,\
                dura_count, prev_dura,\
                rep_count, prev_token, notes, \
                inputs, inputs_dur, h_tm1, c_tm1, \
                x1,x2,x3,a1,a2,a3,\
                pred: i < self.sequence_length,
            body=_d_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),self.sequence_length,tf.constant(0,dtype=tf.int32), self.d0, tf.constant(0,dtype=tf.int32),tf.mod(12+self.p0-self.start_chordkey,12),
                self.chordKeys_onehot[0],self.chordNotes[0],
                tf.constant(1,dtype=tf.int32),self.d0,self.x_dur,
                tf.constant(1,dtype=tf.int32),self.start_dura,
                tf.constant(1,dtype=tf.int32),self.p0,self.x,
                ta_emb_real_x, ta_emb_real_x_dur,self.d_h0,self.d_c0, 
                self.p1,self.p2,self.p3,self.d1,self.d2,self.d3,
                d_real_predictions))
        self.d_real_predictions = tf.reshape(
                self.d_real_predictions.stack(),
                [self.sequence_length])
        self.d_real_predictions_out = tf.identity(self.d_real_predictions,name="d_real_predictions_out")

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

        emb_x_dur = tf.gather(self.g_embeddings_dur, self.x_dur)
        ta_emb_x_dur = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x_dur = ta_emb_x_dur.unstack(emb_x_dur)

        # pretrain recurrence


        def _newpretrain_recurrence(ii, seqindex, beat_count, pitch_count, prev_interval, prev_pitch_chord,
            chordkey_vec, chordnote_vec,
            a_count, prev_a,
            dura_count, prev_dura,
            rep_count, prev_token,
            x_t,a_t, dura_t, h_tm1, c_tm1,
            x1,x2,x3,a1,a2,a3,
            g_predictions,g_predictions_dur): 

            #beatVec = tf.one_hot(beat_count,48,1.0,0.0)
            timeTillEnd = self.chord_length - ii -1 -1 # -1 to boost seqindex, -1 for onehot indexing
            timeTillEndVec = tf.reshape(tf.one_hot(timeTillEnd,self.endCount,on_value=1.0,off_value=0.0), [self.endCount])

            # Feed pitch inputs to input GRU layer of pitch RNN
            rchordnote_vec = tf.reshape(tf.to_float(chordnote_vec), [12])
            rchordkey_vec = tf.reshape(tf.gather(self.g_embeddings_chord,chordkey_vec), [self.emb_dim_chord])
            rrep_count = tf.reshape(tf.to_float(rep_count), [1])
            ra_count = tf.reshape(tf.to_float(a_count), [1])
            rdura_count = tf.reshape(tf.to_float(dura_count),[1])
            rprev_interval = tf.reshape(tf.to_float(prev_interval),[1])
            rprev_pitch_chord = tf.reshape(tf.one_hot(prev_pitch_chord,12,on_value=1.0,off_value=0.0),[12])
            #rblock_input = tf.reshape(tf.to_float(block_input), [self.hidden_dim_b])
            rx_t = tf.reshape(x_t, [self.emb_dim])
            ra_t = tf.reshape(a_t, [self.emb_dim_dur])
            rx1_t = tf.reshape(tf.gather(self.g_embeddings, x1), [self.emb_dim])
            ra1_t = tf.reshape(tf.gather(self.g_embeddings_dur, a1), [self.emb_dim_dur])
            rx2_t = tf.reshape(tf.gather(self.g_embeddings, x2), [self.emb_dim])
            ra2_t = tf.reshape(tf.gather(self.g_embeddings_dur, a2), [self.emb_dim_dur])
            rx3_t = tf.reshape(tf.gather(self.g_embeddings, x3), [self.emb_dim])
            ra3_t = tf.reshape(tf.gather(self.g_embeddings_dur, a3), [self.emb_dim_dur])
            rdura_t = tf.reshape(dura_t,[self.emb_dim_dura])
            #rbeatVec = tf.reshape(beatVec, [48])

            rc_tm1 = tf.reshape(c_tm1, [self.hidden_dim, 1])
            rh_tm1 = tf.reshape(h_tm1, [self.hidden_dim, 1])
            inputs_recurr = tf.reshape(
                tf.concat([rchordnote_vec,rchordkey_vec,rrep_count,ra_count,rdura_count,rprev_interval,rprev_pitch_chord,timeTillEndVec,rx_t,rx1_t,rx2_t,rx3_t,ra_t,ra1_t,ra2_t,ra3_t,rdura_t],
                #tf.concat([rhigh,rlow,rchordnote_vec,rchordkey_vec,rrep_count,ra_count,rdura_count,rprev_interval,rprev_pitch_chord,rblock_input,rx_t,ra_t,rdura_t,rbeatVec],
                    0), 
                [self.num_inputs_block,1])

            firstH, firstC = self.g_recurrent_unit(rh_tm1,inputs_recurr,rc_tm1)
            firstH = tf.reshape(firstH, [self.hidden_dim])
            firstC = tf.reshape(firstC, [self.hidden_dim])

            # Feed output to softmax unit to get next predicted token
            o_t = self.g_output_unit(self.g_embeddings, self.num_emb, self.hidden_dim, firstH)
            next_token = self.x[seqindex]
            x_tp1 = ta_emb_x.read(seqindex)

            oa_t = self.g_output_unit_dur(self.g_embeddings_dur, self.num_emb_dur, self.hidden_dim, firstH)
            next_token_dur = self.x_dur[seqindex]
            a_tp1 = ta_emb_x_dur.read(seqindex)
            next_token_dura = tf.mod(next_token_dur+48-beat_count-1,tf.constant(48,dtype=tf.int32))
            dura_tp1 = tf.gather(self.g_embeddings_dura,next_token_dura)

            g_predictions = g_predictions.write(seqindex, o_t)
            g_predictions_dur = g_predictions_dur.write(seqindex,oa_t)

            newPitch = tf.cond(tf.equal(next_token,tf.constant(self.REST_VAL,dtype=tf.int32)), lambda: pitch_count, lambda: next_token )
            cpitch = tf.cond(tf.equal(next_token,tf.constant(self.REST_VAL,dtype=tf.int32)), lambda: tf.constant(0,dtype=tf.int32), lambda: tf.mod(12+next_token-chordkey_vec,12))
            newInterval = newPitch - pitch_count
            newii = ii + next_token_dura+1 # BEAT POSITION / DURATION ENCODING
            #newii = ii + 1 # TIMESTEP ENCODING
            nextckeysonehot = self.chordKeys_onehot[tf.minimum(newii,self.chord_length-1)]
            nextcnotes = self.chordNotes[tf.minimum(newii,self.chord_length-1)]

            return newii, seqindex+1, next_token_dur, newPitch, newInterval, cpitch,\
                nextckeysonehot,nextcnotes,\
                tf.multiply(a_count,tf.to_int32(tf.equal(prev_a,next_token_dur)))+1,next_token_dur, \
                tf.multiply(dura_count,tf.to_int32(tf.equal(prev_dura,next_token_dura)))+1,next_token_dura, \
                tf.multiply(rep_count,tf.to_int32(tf.equal(prev_token,next_token)))+1,next_token, \
                x_tp1, a_tp1, dura_tp1, firstH, firstC, \
                next_token,x1,x2, next_token_dur,a1,a2,\
                g_predictions,g_predictions_dur

        ii, seqindex, beat_count, pitch_count, prev_interval, prev_pitch_chord,\
        chordkey_vec, chordnote_vec,\
        a_count, prev_a,\
        dura_count,prev_dura,\
        rep_count, prev_token,\
        x_t,a_t,dura_t, h_tm1, c_tm1,\
        x1,x2,x3,a1,a2,a3,\
        self.g_predictions,self.g_predictions_dur= control_flow_ops.while_loop(
            cond = lambda \
                ii, seqindex, beat_count, pitch_count, prev_interval, prev_pitch_chord,\
                chordkey_vec, chordnote_vec,\
                a_count, prev_a,\
                dura_count,prev_dura,\
                rep_count, prev_token,\
                x_t,a_t,dura_t, h_tm1, c_tm1,\
                x1,x2,x3,a1,a2,a3,\
                g_predictions, g_predictions_dur: seqindex < self.sequence_length,
            body = _newpretrain_recurrence,
            loop_vars = (
                tf.constant(0, dtype=tf.int32),tf.constant(0, dtype=tf.int32),self.d0,self.p0,tf.constant(0,dtype=tf.int32),tf.mod(12+self.p0-self.start_chordkey,12),
                self.chordKeys_onehot[0],self.chordNotes[0],
                tf.constant(1,dtype=tf.int32),self.d0,
                tf.constant(1,dtype=tf.int32),self.start_dura,
                tf.constant(1,dtype=tf.int32),self.p0,
                tf.gather(self.g_embeddings, self.p0),tf.gather(self.g_embeddings_dur, self.d0),tf.gather(self.g_embeddings_dura,self.start_dura), self.h0, self.c0,
                self.p1,self.p2,self.p3,self.d1,self.d2,self.d3,
                g_predictions,g_predictions_dur,
                )
            )

        self.g_predictions = tf.reshape(
                self.g_predictions.stack(),
                [self.sequence_length, self.num_emb])
        self.g_predictions_dur = tf.reshape(
                self.g_predictions_dur.stack(),
                [self.sequence_length, self.num_emb_dur])

        # TODO JUMPPOINT
        # calculate discriminator loss
        self.d_gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.d_gen_predictions_out, labels=tf.zeros([self.gen_sequence_length])),name="d_gen_loss")
        self.d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.d_real_predictions_out, labels=tf.ones([self.sequence_length])),name="d_real_loss")

        # calculate generator rewards and loss
        decays = tf.exp(tf.log(self.reward_gamma) * tf.to_float(tf.range(self.gen_sequence_length)))
        rewards = tf.cumsum(decays * tf.sigmoid(self.d_gen_predictions_out),
                                    reverse=True)
        #zero_pads = tf.zeros([self.max_sequence_length - self.sequence_length],tf.float32)
        r_div = tf.div(rewards, tf.cumsum(decays, reverse=True))
        expected_reward_short = tf.slice(self.expected_reward,[0],[self.gen_sequence_length])
        normalized_rewards = r_div - expected_reward_short #\
            #tf.concat([r_div, zero_pads], 0) - self.expected_reward

        self.reward_loss = tf.reduce_mean(normalized_rewards ** 2)
        
        # TODO: ADD dur TO LOSS
        self.g_loss = \
            -tf.reduce_mean(tf.log(tf.clip_by_value(self.gen_o,1e-7,1e7)) * normalized_rewards) \
            -tf.reduce_mean(tf.log(tf.clip_by_value(self.gen_o_dur,1e-7,1e7)) * normalized_rewards)

        # pretraining loss
        self.pretrain_loss = \
            (-tf.reduce_sum(
                tf.one_hot(tf.to_int64(self.x),
                           self.num_emb, on_value=1.0, off_value=0.0) * tf.log(tf.clip_by_value(self.g_predictions,1e-7,1e7)))
             / tf.to_float(self.sequence_length)) \
            + (-tf.reduce_sum(
                tf.one_hot(tf.to_int64(self.x_dur),
                           self.num_emb_dur, on_value=1.0, off_value=0.0) * tf.log(tf.clip_by_value(self.g_predictions_dur,1e-7,1e7)))
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

    def generate(self, session,lengths,chordkeys,chordkeys_onehot,chordnotes,sequence_length,n0,n1,n2,n3):
        start_dura = n0[4]
        start_chordkey = n0[3]
        p0 = n0[0]
        p1 = n1[0]
        p2 = n2[0]
        p3 = n3[0]
        d0 = n0[1]
        d1 = n1[1]
        d2 = n2[1]
        d3 = n3[1]
        outputs = session.run(
                [self.gen_x_out, self.gen_x_dur_out],
                feed_dict={self.lengths: lengths, self.samples: np.random.uniform(size=len(chordkeys_onehot)), self.samples_dur: np.random.uniform(size=len(chordkeys_onehot)),
                           self.chordKeys: chordkeys, self.chordKeys_onehot: chordkeys_onehot, self.chordNotes: chordnotes,
                           self.sequence_length: sequence_length,
                           self.p0:p0,self.p1:p1,self.p2:p2,self.p3:p3,
                           self.d0:d0,self.d1:d1,self.d2:d2,self.d3:d3, 
                           self.start_chordkey:start_chordkey, self.start_dura:start_dura})
        return outputs

    def train_g_step(self, session,lengths,chordkeys,chordkeys_onehot,chordnotes,sequence_length,n0,n1,n2,n3):
        start_dura = n0[4]
        start_chordkey = n0[3]
        p0 = n0[0]
        p1 = n1[0]
        p2 = n2[0]
        p3 = n3[0]
        d0 = n0[1]
        d1 = n1[1]
        d2 = n2[1]
        d3 = n3[1]
        outputs = session.run(
                [self.g_updates, self.reward_updates, self.g_loss,
                 self.expected_reward, self.gen_x, self.gen_x_dur],
                feed_dict={self.lengths: lengths, self.samples: np.random.uniform(size=len(chordkeys_onehot)), self.samples_dur: np.random.uniform(size=len(chordkeys_onehot)),
                           self.chordKeys:chordkeys, self.chordKeys_onehot:chordkeys_onehot, self.chordNotes: chordnotes,
                           self.sequence_length: sequence_length,
                           self.p0:p0,self.p1:p1,self.p2:p2,self.p3:p3,
                           self.d0:d0,self.d1:d1,self.d2:d2,self.d3:d3, 
                           self.start_chordkey:start_chordkey, self.start_dura:start_dura})
        return outputs

    def train_d_gen_step(self, session,lengths,chordkeys,chordkeys_onehot,chordnotes,sequence_length,n0,n1,n2,n3):
        start_dura = n0[4]
        start_chordkey = n0[3]
        p0 = n0[0]
        p1 = n1[0]
        p2 = n2[0]
        p3 = n3[0]
        d0 = n0[1]
        d1 = n1[1]
        d2 = n2[1]
        d3 = n3[1]
        outputs = session.run(
                [self.d_gen_updates, self.d_gen_loss],
                feed_dict={self.lengths: lengths, self.samples: np.random.uniform(size=len(chordkeys_onehot)), self.samples_dur: np.random.uniform(size=len(chordkeys_onehot)),
                           self.chordKeys: chordkeys, self.chordKeys_onehot: chordkeys_onehot, self.chordNotes: chordnotes,
                           self.sequence_length: sequence_length,
                           self.p0:p0,self.p1:p1,self.p2:p2,self.p3:p3,
                           self.d0:d0,self.d1:d1,self.d2:d2,self.d3:d3, 
                           self.start_chordkey:start_chordkey, self.start_dura:start_dura})
        return outputs

    def train_d_real_step(self, session, lengths,x, x_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3):
        start_dura = n0[4]
        start_chordkey = n0[3]
        p0 = n0[0]
        p1 = n1[0]
        p2 = n2[0]
        p3 = n3[0]
        d0 = n0[1]
        d1 = n1[1]
        d2 = n2[1]
        d3 = n3[1]
        outputs = session.run([self.d_real_updates, self.d_real_loss],
                              feed_dict={self.x: x, self.x_dur: x_dur,
                                         self.lengths: lengths, self.samples: np.random.uniform(size=len(chordkeys_onehot)), self.samples_dur: np.random.uniform(size=len(chordkeys_onehot)),
                                         self.chordKeys:chordkeys, self.chordKeys_onehot:chordkeys_onehot,self.chordNotes:chordnotes, self.lows:low, self.highs:high,
                                         self.sequence_length: sequence_length,
                                         self.p0:p0,self.p1:p1,self.p2:p2,self.p3:p3,
                                         self.d0:d0,self.d1:d1,self.d2:d2,self.d3:d3,
                                         self.start_chordkey:start_chordkey, self.start_dura:start_dura})
        return outputs

    def pretrain_step(self, session, lengths,x, x_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3):
        start_dura = n0[4]
        start_chordkey = n0[3]
        p0 = n0[0]
        p1 = n1[0]
        p2 = n2[0]
        p3 = n3[0]
        d0 = n0[1]
        d1 = n1[1]
        d2 = n2[1]
        d3 = n3[1]
        outputs = session.run([self.pretrain_updates, self.pretrain_loss, self.g_predictions, self.g_predictions_dur],
                              feed_dict={self.x: x, self.x_dur: x_dur,
                                         self.lengths: lengths, self.samples: np.random.uniform(size=len(chordkeys_onehot)), self.samples_dur: np.random.uniform(size=len(chordkeys_onehot)),
                                         self.chordKeys:chordkeys, self.chordKeys_onehot:chordkeys_onehot,self.chordNotes:chordnotes, self.lows:low, self.highs:high,
                                         self.sequence_length: sequence_length,
                                         self.p0:p0,self.p1:p1,self.p2:p2,self.p3:p3,
                                         self.d0:d0,self.d1:d1,self.d2:d2,self.d3:d3,
                                         self.start_chordkey:start_chordkey, self.start_dura:start_dura})
        return outputs

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1, dtype = tf.float32)

    def init_vector(self, shape, value = 0):
        return tf.constant(value, shape=shape,dtype=tf.float32)

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
        b_class = tf.Variable(self.init_vector([1]))
        params.extend([W_class, b_class])
        def unit(h_t):
            return b_class + tf.matmul(W_class, tf.reshape(h_t, [self.hidden_dim, 1]))
        return unit

    def create_classifier_unit_d(self, params):
        W_class = tf.Variable(self.init_matrix([1, self.hidden_dim_d]))
        b_class = tf.Variable(self.init_vector([1]))
        params.extend([W_class, b_class])
        def unit(h_t):
            return b_class + tf.matmul(W_class, tf.reshape(h_t, [self.hidden_dim_d, 1]))
        return unit

    def d_optimizer(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)

    def g_optimizer(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)


class GRU(RNN):

    def create_gru_layer(self,hidden_dim, num_features, params):
        W_r = tf.Variable(self.init_matrix([hidden_dim, num_features]))
        W_z = tf.Variable(self.init_matrix([hidden_dim, num_features]))
        W_h = tf.Variable(self.init_matrix([hidden_dim, num_features]))
        U_r = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        U_z = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))
        U_h = tf.Variable(self.init_matrix([hidden_dim, hidden_dim]))

        params.extend([
            W_r, W_z, W_h,
            U_r, U_z, U_h])

        def unit(h,x):
            r = tf.sigmoid(tf.matmul(W_r,x)+tf.matmul(U_r,h))
            z = tf.sigmoid(tf.matmul(W_z,x)+tf.matmul(U_z,h))
            h_tilda = tf.tanh(tf.matmul(W_h,x)+tf.matmul(U_h,r*h))
            h_t = (1 - z) * h + z * h_tilda
            return h_t

        return unit

    def create_lstm_layer(self, hidden_dim, num_features, params):
        b_f = tf.Variable(self.init_vector([hidden_dim,1],value=1))
        b_i = tf.Variable(self.init_vector([hidden_dim,1],value=-1))
        b_c = tf.Variable(self.init_vector([hidden_dim,1],value=-1))
        b_o = tf.Variable(self.init_vector([hidden_dim,1],value=-1))
        W_f = tf.Variable(self.init_matrix([hidden_dim, num_features]))
        W_i = tf.Variable(self.init_matrix([hidden_dim, num_features]))
        W_c = tf.Variable(self.init_matrix([hidden_dim, num_features]))
        W_o = tf.Variable(self.init_matrix([hidden_dim, num_features]))
        
        params.extend([
            b_f,W_f,
            b_i,W_i,
            b_c,W_c,
            b_o,W_o])

        def unit(h,x,c):
            g_in = tf.concat([x,h,c],0)
            f = tf.sigmoid(tf.matmul(W_f,g_in)+b_f)
            i = tf.sigmoid(tf.matmul(W_i,g_in)+b_i)
            c_tilda = tf.tanh(tf.matmul(W_c,g_in)+b_c)
            c_t = f*c+i*c_tilda

            g_out = tf.concat([x,h,c_t],0)
            o = tf.sigmoid(tf.matmul(W_o,g_out)+b_o)
            h_t = o*tf.tanh(c_t)
            return h_t,c_t

        return unit
