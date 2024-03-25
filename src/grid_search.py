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

def train(training_set, dev_set, self_loop, use_final, use_unigram, use_bigram, use_silence, log_file=None):
    if log_file is None:
        log_file = f"./new_logs/self_loop={self_loop}|use_final={use_final}|use_unigram={use_unigram}|use_bigram={use_bigram}|use_silence={use_silence}.txt"

        if os.path.exists(log_file):
            print(f"SKIPPED {log_file}")
            return 1e10, 1e10

    log_output = []

    unigram_probs = compute_unigram_probs(dev_set) if use_unigram else None
    final_probs = compute_final_probs(dev_set) if use_final else None
    bigram_probs = compute_bigram_probs(dev_set) if use_bigram else None

    perplexity_bigrams = compute_bigram_probs(dev_set)
    
    f = generate_sequence_wfst(string, self_loop_prob=self_loop, unigram_probs=unigram_probs, use_sil=use_silence, final_probs=final_probs, bigram_probs=bigram_probs)

    wav_files = 0
    total_errors, total_words = 0, 0
    decode_times = []
    backtrace_times = []
    insertion_errors = 0
    deletion_errors = 0
    sub_errors = 0
    total_perplexity = 0
    for wav_file in training_set:
        wav_files += 1

        decoder = MyViterbiDecoder(f, wav_file, None, None)

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
    average_decode = f"Average decode() Time: {sum(decode_times) / wav_files}"
    average_backtrace = f"Average backtrace() Time: {sum(backtrace_times) / wav_files}"
    fst_num_states = f"FST Number of States: {f.num_states()}"
    fst_num_arcs = f"FST Number of Arcs: {get_num_arcs(f)}"
    total_ins_errors = f"Number of Insertion Errors: {insertion_errors}"
    total_del_errors = f"Number of Deletion Errors: {deletion_errors}"
    total_sub_errors = f"Number of Substitution Errors: {sub_errors}"

    for log in [final_wer, final_pp, average_decode, average_backtrace, fst_num_states, fst_num_arcs, total_ins_errors, total_del_errors, total_sub_errors]:
        log_output.append(log + "\n")

    with open(log_file, "w") as f:
        f.writelines(log_output)

    print("TRAINING COMPLETE")

    return (total_errors / total_words), (total_perplexity / len(training_set))

if __name__ == '__main__':
    # Grid search parameters
    self_loop_probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    use_final_probs = [False, True]
    use_unigram = [False, True]
    use_bigram = [False, True]
    use_silence_hmm = [None, 'linear-3', 'linear-5', 'ergodic-5']

    best_wer = 1e10
    best_pp = 1e10
    best_performing_wer = ""
    best_performing_pp = ""

    training_set, dev_set = split_data()

    # Create the lexicon
    string = "peter piper picked a peck of pickled peppers where's the peck of pickled peppers peter piper picked"
    lex = parse_lexicon('lexicon.txt')
    word_table, phone_table, state_table = generate_symbol_tables(lex)
    
    for self_loop in self_loop_probs:
        for use_final in use_final_probs:
            for bigram in use_bigram:
                for unigram in use_unigram:
                    for use_silence in use_silence_hmm:
                        out_wer, out_perp = train(training_set, dev_set, self_loop, use_final, unigram, bigram, use_silence)

                        if out_wer < best_wer:
                            best_performing_wer = f"{self_loop}|{use_final}|{bigram}|{unigram}|{use_silence}"
                            best_wer = out_wer
                        if out_perp < best_pp:
                            best_performing_pp = f"{self_loop}|{use_final}|{bigram}|{unigram}|{use_silence}"
                            best_pp = out_perp

    # Write best performing
    with open("./logs/best_performing.txt", "w") as f:
        f.write(f"Best Performing WER: {best_performing_wer}")
        f.write(f"Best Performing PP: {best_performing_pp}")
