from __future__ import print_function

__doc__ = """Training utility functions."""

import numpy as np
import random

def constructPrintStrings(n0,n1,n2,n3, lengths, noteseq, durseq):
    notestr ="["+ str(n0[0]) + ", " + str(n1[0]) + ", " + str(n2[0]) + ", " + str(n3[0])
    durstr ="["+ str(n0[1]) + ", " + str(n1[1]) + ", " + str(n2[1]) + ", " + str(n3[1])
    noteindex = 0
    for l in lengths:
        notestr += ", ("
        durstr += ", ("
        first = True
        for _ in range(l):
            if not first:
                notestr += ", "
                durstr += ", "
            first = False
            notestr += str(noteseq[noteindex])
            durstr += str(durseq[noteindex])
            noteindex += 1
        notestr += ")"
        durstr += ")"
    notestr += "]"
    durstr += "]"
    return notestr,durstr

def train_epoch(sess, trainable_model, num_iter,
                proportion_supervised, g_steps, d_steps,
                next_sequence, next_sequence_lengths, sequences, durseqs, chordseqs, lows,highs,spseq,
                words=None,
                proportion_generated=0.5,
                skipDiscriminator=False,
                skipGenerator=False,note_adjust=0, ii=0,direction=1):
    """Perform training for model.

    sess: tensorflow session
    trainable_model: the model
    num_iter: number of iterations
    proportion_supervised: what proportion of iterations should the generator
        be trained in a supervised manner (rather than trained via discriminator)
    g_steps: number of generator training steps per iteration
    d_steps: number of discriminator training steps per iteration
    next_sequence: function that returns a groundtruth sequence
    words:  array of words (to map indices back to words)
    proportion_generated: what proportion of steps for the discriminator
        should be on artificially generated data

    """
    supervised_g_losses = [0]  # we put in 0 to avoid empty slices
    unsupervised_g_losses = [0]  # we put in 0 to avoid empty slices
    d_losses = [0]
    g_loss = None
    d_loss = None
    expected_rewards = [[0] * trainable_model.max_sequence_length]
    supervised_gen_x = None
    supervised_gen_x_dur = None
    supervised_chord_key = None
    supervised_chord_key_onehot = None
    supervised_chord_notes = None
    supervised_n0 = None
    supervised_n1 = None
    supervised_n2 = None
    supervised_n3 = None
    supervised_lengths = None
    unsupervised_gen_x = None
    unsupervised_gen_x_dur = None
    unsupervised_chord_key = None
    unsupervised_chord_key_onehot = None
    unsupervised_chord_notes = None
    unsupervised_n0 = None
    unsupervised_n1 = None
    unsupervised_n2 = None
    unsupervised_n3 = None
    unsupervised_lengths = None
    actual_seq = None
    actual_seq_dur = None
    print('running %d iterations with %d g steps and %d d steps' % (num_iter, g_steps, d_steps))
    print('of the g steps, %.2f will be supervised' % proportion_supervised)
    for it in range(num_iter):
        if not skipGenerator:
            for _ in range(g_steps):
                if random.random() < proportion_supervised:
                    ii,direction,seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3 = next_sequence(ii,direction,sequences,durseqs,chordseqs,lows,highs,spseq)
                    supervised_chord_key = chordkeys
                    supervised_chord_key_onehot = chordkeys_onehot
                    supervised_chord_notes = chordnotes
                    supervised_n0 = n0
                    supervised_n1 = n1
                    supervised_n2 = n2
                    supervised_n3 = n3
                    actual_seq = seq
                    actual_seq_dur = seq_dur
                    lengths = next_sequence_lengths(sequence_length)
                    supervised_lengths = lengths
                    _, g_loss, g_pred, g_pred_dur = trainable_model.pretrain_step(sess,lengths, seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3)
                    supervised_g_losses.append(g_loss)
                    supervised_gen_x = np.argmax(g_pred, axis=1)
                    supervised_gen_x_dur = np.argmax(g_pred_dur, axis=1)
                else:
                    ii,direction,seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3 = next_sequence(ii,direction,sequences,durseqs,chordseqs,lows,highs,spseq)
                    unsupervised_chord_key = chordkeys
                    unsupervised_chord_key_onehot = chordkeys_onehot
                    unsupervised_chord_notes = chordnotes
                    unsupervised_n0 = n0
                    unsupervised_n1 = n1
                    unsupervised_n2 = n2
                    unsupervised_n3 = n3
                    lengths = next_sequence_lengths(sequence_length)
                    unsupervised_lengths = lengths
                    _, _, g_loss, expected_reward, unsupervised_gen_x, unsupervised_gen_x_dur = \
                        trainable_model.train_g_step(sess,lengths,chordkeys,chordkeys_onehot,chordnotes,sequence_length,n0,n1,n2,n3)
                    expected_rewards.append(expected_reward)
                    unsupervised_g_losses.append(g_loss)
        if not skipDiscriminator:
            for _ in range(d_steps):
                if random.random() < proportion_generated:
                    ii,direction,seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3 = next_sequence(ii,direction,sequences,durseqs,chordseqs,lows,highs,spseq)
                    lengths = next_sequence_lengths(sequence_length)
                    _, d_loss = trainable_model.train_d_real_step(sess,lengths, seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3)
                else:
                    ii,direction,seq, seq_dur,chordkeys,chordkeys_onehot,chordnotes,low,high,sequence_length,n0,n1,n2,n3 = next_sequence(ii,direction,sequences,durseqs,chordseqs,lows,highs,spseq)
                    lengths = next_sequence_lengths(sequence_length)
                    _, d_loss = trainable_model.train_d_gen_step(sess,lengths, chordkeys,chordkeys_onehot,chordnotes,sequence_length,n0,n1,n2,n3)
                d_losses.append(d_loss)

    print('epoch statistics:')
    print('>>>> discriminator loss:', np.mean(d_losses))
    print('>>>> generator loss:', np.mean(supervised_g_losses), np.mean(unsupervised_g_losses))
    print('>>>> actual melody:')
    actual_seq_print = None if actual_seq==None else [x-note_adjust for x in actual_seq]
    if actual_seq_print == None:
        print(None)
        print(None)
    else:
        notestr,durstr = constructPrintStrings(supervised_n0,supervised_n1,supervised_n2,supervised_n3,supervised_lengths,actual_seq_print,actual_seq_dur)
        print(notestr)
        print(durstr)
    print('>>>> sampled generations (supervised, unsupervised):',)
    sup_gen_x = [words[x]- note_adjust if words else x- note_adjust for x in supervised_gen_x] if supervised_gen_x is not None else None
    sup_gen_x_dur = [words[x] if words else x for x in supervised_gen_x_dur] if supervised_gen_x_dur is not None else None
    if sup_gen_x == None:
        print(None)
        print(None)
    else:
        notestr,durstr = constructPrintStrings(supervised_n0,supervised_n1,supervised_n2,supervised_n3,supervised_lengths,sup_gen_x,sup_gen_x_dur)
        print(notestr)
        print(durstr)
    unsup_gen_x = [words[x]- note_adjust if words else x- note_adjust for x in unsupervised_gen_x] if unsupervised_gen_x is not None else None
    unsup_gen_x_dur = [words[x] if words else x for x in unsupervised_gen_x_dur] if unsupervised_gen_x_dur is not None else None
    if unsup_gen_x == None:
        print(None)
        print(None)
    else:
        notestr,durstr = constructPrintStrings(unsupervised_n0,unsupervised_n1,unsupervised_n2,unsupervised_n3,unsupervised_lengths,unsup_gen_x,unsup_gen_x_dur)
        print(notestr)
        print(durstr)
    print('>>>> expected rewards:', np.mean(expected_rewards, axis=0))
    return ii,direction,g_loss,d_loss,actual_seq_print, actual_seq_dur, sup_gen_x, sup_gen_x_dur, unsup_gen_x, unsup_gen_x_dur, \
        supervised_chord_key, supervised_chord_key_onehot, supervised_chord_notes,supervised_n0,supervised_n1,supervised_n2,supervised_n3,\
        unsupervised_chord_key, unsupervised_chord_key_onehot, unsupervised_chord_notes,unsupervised_n0,unsupervised_n1,unsupervised_n2,unsupervised_n3
