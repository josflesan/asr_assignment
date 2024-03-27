#!/usr/bin/env python3

import argparse
import sys
import json
import glob
import timeit
import os
import wer
import numpy as np
import observation_model
import openfst_python as fst
from helper_functions import parse_lexicon, generate_symbol_tables
from fst import *
from decoder import *
from utils import *

def get_num_arcs(f):
    return sum(1 + f.num_arcs(s) for s in f.states())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--self-arc-prob', type=float)
    parser.add_argument('--use-final-prob', type=float)
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

def split_data(split=0.5):
    all_data = []
    
    # Cycle through number of wave files and save each to a list
    for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
        all_data.append(wav_file)

    return all_data[:int(len(all_data) * split)], all_data[int(len(all_data) * split):]

def train(training_set, dev_set, self_loop=0.9, use_final=False, use_unigram=True, use_bigram=True, use_silence='linear-5', threshold=1e10, beam_size=1e10, use_tree=False, log_file=None):
    if log_file is None:
        log_file = f"./baseline_tree_search/threshold={threshold}|beam_size={beam_size}.txt"

        if os.path.exists(log_file):
            print(f"SKIPPED {log_file}")
            return 1e10, 1e10

    print(f"RUNNING threshold={threshold} | beam size = {beam_size}")

    log_output = []
    string = "peter piper picked a peck of pickled peppers where's the peck of pickled peppers peter piper picked"

    unigram_probs = compute_unigram_probs(dev_set) if use_unigram else None
    final_probs = compute_final_probs(dev_set) if use_final else None
    bigram_probs = compute_bigram_probs(dev_set) if use_bigram else None

    perplexity_bigrams = compute_bigram_probs(dev_set)
    
    if use_tree:
        f = generate_tree_wfst(string, self_loop_prob=0.1)
    else:
        f = generate_sequence_wfst(string, self_loop_prob=self_loop, unigram_probs=unigram_probs, use_sil=use_silence, final_probs=final_probs, bigram_probs=bigram_probs)

    wav_files = 0
    total_errors, total_words = 0, 0
    decode_times = []
    backtrace_times = []
    insertion_errors = 0
    deletion_errors = 0
    sub_errors = 0
    total_perplexity = 0

    total_no_finish = 0
    for wav_file in training_set:
        wav_files += 1

        decoder = MyViterbiDecoder(f, wav_file, threshold, beam_size)

        decode_time = timeit.timeit(lambda: decoder.decode(), number=1)
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
        print(compute_perplexity(words.split(), perplexity_bigrams))

        # Format log string and save
        log_string = f"Wav File: {wav_files}/{len(training_set)} | Errors: {error_counts} | Transcription Length: {word_count}"
        log_output.append(log_string + "\n")

        print(log_string)

    final_wer = f"Total WER: {total_errors / total_words}"
    final_pp = f"Avg Perplexity: {total_perplexity / len(training_set)}"
    no_finish = f"Num No Finish: {total_no_finish}"
    average_decode = f"Average decode() Time: {sum(decode_times) / wav_files}"
    average_backtrace = f"Average backtrace() Time: {sum(backtrace_times) / wav_files}"
    fst_num_states = f"FST Number of States: {f.num_states()}"
    fst_num_arcs = f"FST Number of Arcs: {get_num_arcs(f)}"
    total_ins_errors = f"Number of Insertion Errors: {insertion_errors}"
    total_del_errors = f"Number of Deletion Errors: {deletion_errors}"
    total_sub_errors = f"Number of Substitution Errors: {sub_errors}"

    for log in [final_wer, final_pp, no_finish, average_decode, average_backtrace, fst_num_states, fst_num_arcs, total_ins_errors, total_del_errors, total_sub_errors]:
        log_output.append(log + "\n")

    with open(log_file, "w") as f:
        f.writelines(log_output)

    print("TRAINING COMPLETE")

    return (total_errors / total_words), (total_perplexity / len(training_set))

if __name__ == '__main__':
    # Grid search parameters
    thresholds = [i for i in range(100, 201, 20)][::-1]
    beam_sizes = [i for i in range(40, 81, 20)][::-1]

    best_wer = 1e10
    best_pp = 1e10
    best_performing_wer = ""
    best_performing_pp = ""

    training_set, dev_set = split_data()

    # Create the lexicon
    lex = parse_lexicon('lexicon.txt')
    word_table, phone_table, state_table = generate_symbol_tables(lex)

    for threshold in thresholds:
        for beam_size in beam_sizes:
            out_wer, out_perp = train(training_set, dev_set, threshold=threshold, beam_size=beam_size, use_tree=True)

            if out_wer < best_wer:
                best_performing_wer = f"{threshold}|{beam_size}"
                best_wer = out_wer
            if out_perp < best_pp:
                best_performing_wer = f"{threshold}|{beam_size}"
                best_pp = out_perp

    # Write best performing
    with open("./baseline_tree_search/best_performing.txt", "w") as f:
        f.write(f"Best Performing WER: {best_performing_wer}")
        f.write(f"Best Performing PP: {best_performing_pp}")

