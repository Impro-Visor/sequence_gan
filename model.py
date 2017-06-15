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
    upper_triangular_ones = tf.constant(
        np.triu(np.ones((length, length))),
        dtype=tf.float32)
    return tf.reshape(
            tf.matmul(upper_triangular_ones,
                      tf.reshape(x, [length, 1])),
            [length])



WHOLE_NOTE = 48
HALF_NOTE = 24
QUARTER_NOTE = 12
EIGHTH_NOTE = 6
SIXTEENTH_NOTE = 3
HALF_TRIPLET = 8
QUARTER_TRIPLET = 4
EIGHTH_TRIPLET = 2

class RNN(object):

    def __init__(self, num_emb, num_emb_attack, emb_dim, emb_dim_attack, hidden_dim, hidden_dim_attack,
                 sequence_length, start_token, start_token_attack,
                 learning_rate=0.01, reward_gamma=0.9):
        self.num_emb = num_emb
        self.num_emb_attack = num_emb_attack
        self.emb_dim = emb_dim
        self.emb_dim_attack = emb_dim_attack
        self.hidden_dim = hidden_dim
        self.hidden_dim_attack = hidden_dim_attack
        self.sequence_length = sequence_length
        self.start_token = tf.constant(start_token, dtype=tf.int32)
        self.start_token_attack = tf.constant(start_token_attack, dtype=tf.int32)
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.reward_gamma = reward_gamma
        self.g_params = []
        self.d_params = []

        self.expected_reward = tf.Variable(tf.zeros([self.sequence_length]))

        numBeatsInMeasure = tf.constant(48,dtype=tf.int32)
        beatsConsideredVec = tf.constant([WHOLE_NOTE,HALF_NOTE,QUARTER_NOTE,EIGHTH_NOTE,SIXTEENTH_NOTE,HALF_TRIPLET,QUARTER_TRIPLET,EIGHTH_TRIPLET])
        self.lenBeatVec = tf.size(beatsConsideredVec)

        with tf.variable_scope('generator'):
            # Embedding matrix for notes
            self.g_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.g_params.append(self.g_embeddings)

            # Embedding matrix for attack bit
            self.g_embeddings_attack = tf.Variable(self.init_matrix([self.num_emb_attack,self.emb_dim_attack]))
            self.g_params.append(self.g_embeddings_attack)

            self.g_recurrent_unit = self.create_recurrent_unit(self.emb_dim, self.hidden_dim, self.g_params)  # maps h_tm1 to h_t for generator
            self.g_recurrent_unit_attack = self.create_recurrent_unit(self.emb_dim_attack, self.hidden_dim_attack, self.g_params)

            self.g_output_unit = self.create_output_unit(self.num_emb, self.emb_dim, self.hidden_dim, self.g_params, self.g_embeddings)  # maps h_t to o_t (output token logits)
            self.g_output_unit_attack = self.create_output_unit(self.num_emb_attack, self.emb_dim_attack, self.hidden_dim_attack, self.g_params, self.g_embeddings_attack)

        with tf.variable_scope('discriminator'):
            # Embedding matrix for notes
            self.d_embeddings = tf.Variable(self.init_matrix([self.num_emb, self.emb_dim]))
            self.d_params.append(self.d_embeddings)

            # Embedding matrix for attack bit
            self.d_embeddings_attack = tf.Variable(self.init_matrix([self.num_emb_attack, self.emb_dim_attack]))
            self.d_params.append(self.d_embeddings_attack)

            self.d_recurrent_unit = self.create_recurrent_unit(self.emb_dim, self.hidden_dim, self.d_params)  # maps h_tm1 to h_t for discriminator
            self.d_recurrent_unit_attack = self.create_recurrent_unit(self.emb_dim_attack, self.hidden_dim_attack, self.d_params)  # maps h_tm1 to h_t for discriminator
            self.d_classifier_unit = self.create_classifier_unit(self.d_params)  # maps h_t to class prediction logits
            self.d_h0 = tf.Variable(self.init_vector([self.hidden_dim]))
            self.d_h0_attack = tf.Variable(self.init_vector([self.hidden_dim_attack]))
            self.d_params.append(self.d_h0)
            self.d_params.append(self.d_h0_attack)

        self.h0 = tf.placeholder(tf.float32, shape=[self.hidden_dim])  # initial random vector for generator
        self.h0_attack = tf.placeholder(tf.float32, shape=[self.hidden_dim_attack]) # initial random vector for generator attack
        self.x = tf.placeholder(tf.int32, shape=[self.sequence_length])  # sequence of indices of true notes, not including start token
        self.x_attack = tf.placeholder(tf.int32, shape=[self.sequence_length])  # sequence of indices of true attacks, not including start token        
        self.samples = tf.placeholder(tf.float32, shape=[self.sequence_length])  # random samples from [0, 1]

        # generator on initial randomness
        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_o_attack = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)
        gen_x_attack = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        samples = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        samples = samples.unstack(self.samples)
        def _g_recurrence(i, 
            a_count, prev_a,
            rep_count, prev_token, 
            x_t,a_t, h_tm1, h_tm1_attack,
            gen_o, gen_x, gen_o_attack, gen_x_attack):
            beat = tf.mod(i,numBeatsInMeasure)+tf.constant(1,dtype=tf.int32)
            beatVec = tf.map_fn(lambda i : tf.cast(tf.equal(tf.mod(beat,i),tf.constant(0,dtype=tf.int32)),tf.float32), beatsConsideredVec, dtype=tf.float32)
            sample = samples.read(i)

            h_t_attack = self.g_recurrent_unit_attack(self.emb_dim_attack, self.hidden_dim_attack, a_t, beatVec, a_count, h_tm1_attack)
            oa_t = self.g_output_unit_attack(self.g_embeddings_attack, self.num_emb_attack, self.hidden_dim_attack, h_t_attack)
            oa_cumsum = _cumsum(oa_t, self.num_emb_attack)
            next_token_attack = tf.to_int32(tf.reduce_min(tf.where(sample < oa_cumsum)))

            doingSustain = tf.equal(tf.constant(1,dtype=tf.int32), next_token_attack)
            h_t = self.g_recurrent_unit(self.emb_dim, self.hidden_dim, x_t, beatVec, rep_count, h_tm1)
            o_t = self.g_output_unit(self.g_embeddings, self.num_emb, self.hidden_dim, h_t)
            o_cumsum = _cumsum(o_t, self.num_emb)  # prepare for sampling
            next_token = tf.to_int32(tf.reduce_min(tf.where(sample < o_cumsum)))   # sample

            x_tp1 = tf.gather(self.g_embeddings, next_token)
            gen_o = gen_o.write(i, tf.gather(o_t, next_token))  # we only need the sampled token's probability
            gen_x = gen_x.write(i, next_token)  # indices, not embeddings

            a_tp1 = tf.gather(self.g_embeddings_attack, next_token_attack)
            gen_o_attack = gen_o_attack.write(i, tf.gather(oa_t, next_token_attack))  # we only need the sampled token's probability
            gen_x_attack = gen_x_attack.write(i, next_token_attack)  # indices, not embeddings

            return i + 1, \
                tf.multiply(a_count,tf.cast(tf.equal(prev_a,next_token_attack),tf.int32))+1,next_token_attack, \
                tf.multiply(rep_count,tf.cast(tf.equal(prev_token,next_token),tf.int32))+1,next_token, \
                x_tp1, a_tp1, h_t, h_t_attack,\
                gen_o, gen_x,gen_o_attack,gen_x_attack

        _, _, _, _, _, _, _, _, _, self.gen_o, self.gen_x, self.gen_o_attack, self.gen_x_attack = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                tf.constant(0,dtype=tf.int32),self.start_token_attack,
                tf.constant(0,dtype=tf.int32),self.start_token,
                tf.gather(self.g_embeddings, self.start_token),tf.gather(self.g_embeddings_attack, self.start_token_attack),self.h0,self.h0_attack, 
                gen_o, gen_x,gen_o_attack,gen_x_attack))

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

        # discriminator on generated and real data: attack vars
        self.gen_x_attack = self.gen_x_attack.stack()
        emb_gen_x_attack = tf.gather(self.d_embeddings_attack, self.gen_x_attack)
        ta_emb_gen_x_attack = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_gen_x_attack = ta_emb_gen_x_attack.unstack(emb_gen_x_attack)

        emb_real_x_attack = tf.gather(self.d_embeddings_attack, self.x_attack)
        ta_emb_real_x_attack = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_real_x_attack = ta_emb_real_x_attack.unstack(emb_real_x_attack)

        # discriminator recurrence

        def _d_recurrence(i,
            a_count, prev_a, attacks,
            rep_count, prev_token, notes, 
            inputs, inputs_attack, h_tm1, h_tm1_attack, pred):
            beat = tf.mod(i,numBeatsInMeasure)+tf.constant(1,dtype=tf.int32)
            beatVec = tf.map_fn(lambda i : tf.cast(tf.equal(tf.mod(beat,i),tf.constant(0,dtype=tf.int32)),tf.float32), beatsConsideredVec, dtype=tf.float32)
            x_t = inputs.read(i)
            next_token = notes[i]
            a_t = inputs_attack.read(i)
            next_a = attacks[i]
            h_t = self.d_recurrent_unit(self.emb_dim, self.hidden_dim, x_t,beatVec, rep_count, h_tm1)
            h_t_attack = self.d_recurrent_unit_attack(self.emb_dim_attack, self.hidden_dim_attack, a_t,beatVec, a_count, h_tm1_attack)
            y_t = self.d_classifier_unit(h_t,h_t_attack)
            pred = pred.write(i, y_t)
            return i + 1, \
                tf.multiply(a_count,tf.cast(tf.equal(prev_a,next_a),tf.int32))+1, next_a,attacks,\
                tf.multiply(rep_count,tf.cast(tf.equal(prev_token,next_token),tf.int32))+1, next_token,notes,\
                inputs, inputs_attack, h_t,h_t_attack, pred

        _, _, _, _, _, _, _, _, _, _, _, self.d_gen_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11: i < self.sequence_length,
            body=_d_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                tf.constant(0,dtype=tf.int32), self.start_token_attack, self.gen_x_attack,
                tf.constant(0,dtype=tf.int32), self.start_token, self.gen_x,
                ta_emb_gen_x, ta_emb_gen_x_attack,self.d_h0,self.d_h0_attack, d_gen_predictions))
        self.d_gen_predictions = tf.reshape(
                self.d_gen_predictions.stack(),
                [self.sequence_length])

        _, _, _, _, _, _, _, _, _, _, _, self.d_real_predictions = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11: i < self.sequence_length,
            body=_d_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                tf.constant(0,dtype=tf.int32),self.start_token_attack,self.x_attack,
                tf.constant(0,dtype=tf.int32),self.start_token,self.x,
                ta_emb_real_x, ta_emb_real_x_attack,self.d_h0,self.d_h0_attack, d_real_predictions))
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

        # supervised pretraining for generator: attack vars
        g_predictions_attack = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)

        emb_x_attack = tf.gather(self.g_embeddings_attack, self.x_attack)
        ta_emb_x_attack = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x_attack = ta_emb_x_attack.unstack(emb_x_attack)

        # pretrain recurrence

        def _pretrain_recurrence(i, 
            a_count,prev_a,
            rep_count, prev_token, 
            x_t, a_t, h_tm1, h_tm1_attack,
            g_predictions, g_predictions_attack):
            beat = tf.mod(i,numBeatsInMeasure)+tf.constant(1,dtype=tf.int32)
            beatVec = tf.map_fn(lambda i : tf.cast(tf.equal(tf.mod(beat,i),tf.constant(0,dtype=tf.int32)),tf.float32), beatsConsideredVec, dtype=tf.float32)
            h_t = self.g_recurrent_unit(self.emb_dim, self.hidden_dim, x_t, beatVec, rep_count, h_tm1)
            h_t_attack = self.g_recurrent_unit_attack(self.emb_dim_attack, self.hidden_dim_attack, a_t, beatVec, a_count, h_tm1_attack)
            o_t = self.g_output_unit(self.g_embeddings, self.num_emb, self.hidden_dim, h_t)
            oa_t = self.g_output_unit_attack(self.g_embeddings_attack, self.num_emb_attack,self.hidden_dim_attack, h_t_attack)
            g_predictions = g_predictions.write(i, o_t)
            g_predictions_attack = g_predictions_attack.write(i, oa_t)
            x_tp1 = ta_emb_x.read(i)
            a_tp1 = ta_emb_x_attack.read(i)

            next_a = self.x_attack[i]
            next_token = self.x[i]
            return i + 1, \
                tf.multiply(a_count,tf.cast(tf.equal(prev_a,next_a),tf.int32))+1, next_a,\
                tf.multiply(rep_count,tf.cast(tf.equal(prev_token,next_token),tf.int32))+1, next_token,\
                x_tp1, a_tp1, h_t, h_t_attack,\
                g_predictions, g_predictions_attack

        _, _, _, _, _, _, _, _, _, self.g_predictions, self.g_predictions_attack = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10: i < self.sequence_length,
            body=_pretrain_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                tf.constant(0,dtype=tf.int32),self.start_token_attack,
                tf.constant(0,dtype=tf.int32),self.start_token,
                tf.gather(self.g_embeddings, self.start_token), tf.gather(self.g_embeddings_attack,self.start_token_attack),self.h0, self.h0_attack,
                g_predictions,g_predictions_attack))

        self.g_predictions = tf.reshape(
                self.g_predictions.stack(),
                [self.sequence_length, self.num_emb])
        self.g_predictions_attack = tf.reshape(
                self.g_predictions_attack.stack(),
                [self.sequence_length, self.num_emb_attack])

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
        normalized_rewards = \
            rewards / _backwards_cumsum(decays, self.sequence_length) - self.expected_reward

        self.reward_loss = tf.reduce_mean(normalized_rewards ** 2)
        
        # TODO: ADD ATTACK TO LOSS
        self.g_loss = \
            -tf.reduce_mean(tf.log(tf.clip_by_value(self.gen_o.stack(),1e-7,1e7)) * normalized_rewards) \
            -tf.reduce_mean(tf.log(tf.clip_by_value(self.gen_o_attack.stack(),1e-7,1e7)) * normalized_rewards)

        # pretraining loss
        self.pretrain_loss = \
            (-tf.reduce_sum(
                tf.one_hot(tf.to_int64(self.x),
                           self.num_emb, 1.0, 0.0) * tf.log(tf.clip_by_value(self.g_predictions,1e-7,1e7)))
             / self.sequence_length) \
            + (-tf.reduce_sum(
                tf.one_hot(tf.to_int64(self.x_attack),
                           self.num_emb_attack, 1.0, 0.0) * tf.log(tf.clip_by_value(self.g_predictions_attack,1e-7,1e7)))
             / self.sequence_length)

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

    def generate(self, session):
        outputs = session.run(
                [self.gen_x, self.gen_x_attack],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dim),
                           self.h0_attack: np.random.normal(size=self.hidden_dim_attack),
                           self.samples: np.random.random(self.sequence_length)})
        return outputs[0]

    def train_g_step(self, session):
        outputs = session.run(
                [self.g_updates, self.reward_updates, self.g_loss,
                 self.expected_reward, self.gen_x, self.gen_x_attack],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dim),
                           self.h0_attack: np.random.normal(size=self.hidden_dim_attack),
                           self.samples: np.random.random(self.sequence_length)})
        return outputs

    def train_d_gen_step(self, session):
        outputs = session.run(
                [self.d_gen_updates, self.d_gen_loss],
                feed_dict={self.h0: np.random.normal(size=self.hidden_dim),
                           self.h0_attack: np.random.normal(size=self.hidden_dim_attack),
                           self.samples: np.random.random(self.sequence_length)})
        return outputs

    def train_d_real_step(self, session, x, x_attack):
        outputs = session.run([self.d_real_updates, self.d_real_loss],
                              feed_dict={self.x: x, self.x_attack: x_attack})
        return outputs

    def pretrain_step(self, session, x, x_attack):
        outputs = session.run([self.pretrain_updates, self.pretrain_loss, self.g_predictions, self.g_predictions_attack],
                              feed_dict={self.x: x, self.x_attack: x_attack,
                                         self.h0_attack: np.random.normal(size=self.hidden_dim_attack),
                                         self.h0: np.random.normal(size=self.hidden_dim)})
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
        W_class_attack = tf.Variable(self.init_matrix([1, self.hidden_dim_attack]))
        b_class = tf.Variable(self.init_vector([1]))
        params.extend([W_class, b_class, W_class_attack])
        def unit(h_t, h_t_attack):
            return b_class + tf.matmul(W_class, tf.reshape(h_t, [self.hidden_dim, 1])) \
                + tf.matmul(W_class_attack, tf.reshape(h_t_attack, [self.hidden_dim_attack,1]))
        return unit

    def d_optimizer(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)

    def g_optimizer(self, *args, **kwargs):
        return tf.train.GradientDescentOptimizer(*args, **kwargs)


class GRU(RNN):

    def create_recurrent_unit(self, emb_dim,hidden_dim, params):
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

            rep_count = tf.reshape(tf.cast(rep_count,tf.float32), [1,1])
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
