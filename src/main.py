#!/usr/bin/env python3

import argparse
import sys
import json
import glob
import timeit
import os
import wer
import observation_model
import openfst_python as fst
from helper_functions import parse_lexicon, generate_symbol_tables
from fst import *
from decoder import *
from utils import *

def get_num_arcs(f):
    return sum(1 + f.num_arcs(s) for s in f.states())

def split_data(split=0.5):
    all_data = []
    
    # Cycle through number of wave files and save each to a list
    for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
        all_data.append(wav_file)

    return all_data[:int(len(all_data) * split)], all_data[int(len(all_data) * split):]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--self-arc-prob', type=float)
    parser.add_argument('--use-final-prob', type=str, choices=["linear-3", "linear-5", "ergodic-5"])
    parser.add_argument('--use-unigram', type=bool)
    parser.add_argument('--use-bigram', type=bool)
    parser.add_argument('--use-silence-state', type=bool)
    parser.add_argument('--silence-state-num', type=int)
    parser.add_argument('--pruning-threshold', type=float)
    parser.add_argument('--pruning-beam', type=int)
    parser.add_argument('--pruning-strategy', choices=["normal"], type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        exit(1)

    args = parser.parse_args()

    if args.config:
        f = open(args.config)
        config = json.load(f)
        f.close()

        for k, v in config.items():
            if k not in args.__dict__ or args.__dict__[k] is None:
                args.__dict__[k] = v

    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print()

    return args

if __name__ == '__main__':
    print(' '.join(sys.argv))
    args = parse_args()

    test_set, dev_set = split_data()

    unigram_probs = compute_unigram_probs(dev_set) if args.use_unigram else None
    bigram_probs = compute_bigram_probs(dev_set) if args.use_bigram else None
    final_probs = compute_final_probs(dev_set) if args.use_final_prob else None

    perplexity_bigrams = compute_bigram_probs(dev_set)

    plot_word_dist()

    # Create the lexicon
    string = "peter piper picked a peck of pickled peppers where's the peck of pickled peppers peter piper picked"
    lex = parse_lexicon('lexicon.txt')
    word_table, phone_table, state_table = generate_symbol_tables(lex)

    f = generate_sequence_wfst(string, self_loop_prob=args.self_arc_prob, unigram_probs=unigram_probs, use_sil=args.use_silence_state, final_probs=final_probs, bigram_probs=bigram_probs)
    display_fst(f, "f.png")

    # Train and Report Metrics
    wav_files = 0
    total_errors, total_words = 0, 0
    decode_times = []
    backtrace_times = []
    insertion_errors = 0
    deletion_errors = 0
    sub_errors = 0
    total_perplexity = 0
    for wav_file in test_set:
        wav_files += 1

        decoder = MyViterbiDecoder(f, wav_file, args.pruning_threshold, args.pruning_beam)
        
        decode_time = timeit.timeit(lambda: decoder.decode(False), number=1)
        backtrace_time = timeit.timeit(lambda: decoder.backtrace(), number=1)
        decode_times.append(decode_time)
        backtrace_times.append(backtrace_time)
        (state_path, words) = decoder.backtrace()

        transcription = read_transcription(wav_file)
        error_counts = wer.compute_alignment_errors(transcription, words)
        word_count = len(transcription.split())

        total_errors += sum(error_counts)
        total_words += word_count

        sub_errors += error_counts[0]
        deletion_errors += error_counts[1]
        insertion_errors += error_counts[2]
        
        total_perplexity += compute_perplexity(words.split(), perplexity_bigrams)

        print(wav_files, error_counts, word_count)

    print(f"Total WER: {total_errors / total_words}")
    print(f"Avg. Perplexity: {total_perplexity / len(test_set)}")
    print(f"Average decode() Time: {sum(decode_times) / wav_files}")
    print(f"Average backtrace() Time: {sum(backtrace_times) / wav_files}")
    print(f"FST Number of States: {f.num_states()}")
    print(f"FST Number of Arcs: {get_num_arcs(f)}")
    print(f"Number of Insertion Errors: {insertion_errors}")
    print(f"Number of Deletion Errors: {deletion_errors}")
    print(f"Number of Substitution Errors: {sub_errors}")
