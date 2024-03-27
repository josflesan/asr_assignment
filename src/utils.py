import os
import math
import glob
import numpy as np
import matplotlib.pyplot as plt
from subprocess import check_call
from IPython.display import Image

def display_fst(f, filename="tmp.png"):
    f.draw('tmp.dot', portrait=True)
    check_call(['dot','-Tpng','-Gdpi=1600','tmp.dot','-o', filename])
    return Image(filename=filename)

def plot_efficiency_heatmap():
    results = {}
    for log_file in glob.glob('tree_search/*.txt'):
        if log_file == "tree_search/best_performing.txt":
            continue

        current_threshold = log_file.strip('tree_search/').split('|')[0].split('=')[1]
        current_beam = log_file.strip('tree_search/').split('|')[1].split('=')[1][:-3]
        result_label = f"{current_threshold}_{current_beam}"

        with open(log_file, "r") as f:
            log_lines = f.readlines()
            # wer = float(log_lines[-10].strip().split(':')[1])
            decode_time = float(log_lines[-7].strip().split(':')[1])
            results[result_label] = decode_time

    # results["MAX_MAX"] = 0.3119

    thresholds = sorted(list(set([label.split("_")[0] for label in results.keys()])))
    beam_sizes = sorted(list(set([label.split("_")[1] for label in results.keys()])))
    
    data = np.zeros((len(thresholds), len(beam_sizes)))
    
    for label, decode_time in results.items():
        threshold, beam_size = label.split("_")

        print(threshold)
        print(beam_size)
        data[thresholds.index(threshold), beam_sizes.index(beam_size)] = decode_time

    plt.figure(figsize=(8, 8))

    plt.imshow(data, cmap='winter_r', interpolation='nearest')
    colorbar = plt.colorbar()

    plt.ylabel("Threshold", fontsize=16)
    plt.xlabel("Beam Size", fontsize=16)
    colorbar.set_label("WER")

    # Add text annotations for WERs
    for i in range(len(data)):
        for j in range(len(data[0])):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='white', fontsize=15)

    plt.xticks(np.arange(len(beam_sizes)), beam_sizes, fontsize=14)
    plt.yticks(np.arange(len(thresholds)), thresholds, fontsize=14)

    plt.savefig('heatmap_tree.png', dpi=500)


def plot_tradeoff():
    results = {}
    for log_file in glob.glob('logs/*.txt'):
        current_threshold = log_file.strip('logs/').split('|')[0].split('=')[1]
        current_beam = log_file.strip('logs/').split('|')[1].split('=')[1][:-4]
        result_label = f"{current_threshold}_{current_beam}"

        with open(log_file, "r") as f:
            log_lines = f.readlines()
            wer = float(log_lines[-10].strip().split(':')[1])
            decode_time = float(log_lines[-7].strip().split(':')[1])
            results[result_label] = (wer, decode_time)

    fig = plt.figure(figsize=(10, 5))
    results["MAX_MAX"] = (0.3119, 1.902)

    plt.scatter([tup[1] for tup in results.values()], [tup[0] for tup in results.values()], label="Normal Solutions")

    non_dominated_labels = []
    non_dominated_front = []
    for label, (decode_time, wer) in results.items():
        dominant = True
        for other_label, (other_time, other_wer) in results.items():
            if label == other_label:
                continue

            if other_time < decode_time and other_wer < wer:
                dominant = False
                break

        if dominant:
            non_dominated_labels.append(label)
            non_dominated_front.append((decode_time, wer))

    sorted_indices = np.argsort(np.array(non_dominated_front)[:, 1])
    non_dominated_front = np.array(non_dominated_front)[sorted_indices].tolist()
    non_dominated_labels = np.array(non_dominated_labels)[sorted_indices].tolist()

    decode_times = [tup[1] for tup in non_dominated_front]
    wers = [tup[0] for tup in non_dominated_front]

    m, b = np.polyfit(decode_times, wers, 1)
    r_squared = (np.corrcoef(decode_times, wers)[0][1])**2

    plt.plot(decode_times, wers, color='orange', marker='^', markersize=12, label="Non-Dominated Solutions")
    plt.plot(decode_times, list(m*np.array(decode_times) + b), color='black', linestyle='dotted', alpha=0.7, label=f"WER = {round(m, 2)}Time + {round(b, 2)} ($R^2$ = {round(r_squared, 2)})")
    
    non_dominated_length = 0
    for i, (label, wer) in enumerate(zip(non_dominated_labels, [wer for wer, _ in non_dominated_front])):
        out_label = f"({label.split('_')[0]}, {label.split('_')[1]})"
        if wer < 1 and label not in ["100_144", "100_168", "200_144", "200_168"]:
            non_dominated_length += 1
            vdisplacement = 0.02
            hdisplacement = 0
            
            if label == "100_192":
                hdisplacement = 0.04
            if label == "200_120":
                vdisplacement = 0.027
            if label == "160_240":
                vdisplacement = 0.025
            if label == "180_240":
                vdisplacement = 0.04
            if label == "200_240":
                vdisplacement = 0.055
            if label == "MAX_MAX":
                hdisplacement = 0.06
                vdisplacement = - 0.04

            plt.text(decode_times[i] + hdisplacement, wers[i] - vdisplacement, out_label, fontsize=8, ha='right', va='top')

    plt.ylabel("WER", fontsize=14)
    plt.xlabel("Decode Time", fontsize=14)
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.legend()
    plt.savefig('scatter.png', dpi=500)

def plot_bigram_dist():
    words = "peter piper picked a peck of pickled peppers where's the peck of pickled peppers peter piper picked"
    total_utterances = 0
    bigram_frequencies = {}
    for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
        transcription = read_transcription(wav_file).split()

        for word1, word2 in zip(transcription[:-1], transcription[1:]):
            bigram = f"{word1}_{word2}"
            if bigram not in bigram_frequencies:
                bigram_frequencies[bigram] = 1
            else:
                bigram_frequencies[bigram] += 1

    bigram_frequencies = sorted(bigram_frequencies.items(), key=lambda x: x[1], reverse=True)
    bigrams, frequencies = zip(*bigram_frequencies)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(bigrams)), frequencies, color='skyblue')
    plt.xticks([])
    plt.yticks(fontsize=14)
    plt.xlabel('Bigrams', fontsize=16)
    plt.ylabel('Frequency Count', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.savefig("bigram_dist.png", dpi=500)

def plot_word_dist():
    words = "peter piper picked a peck of pickled peppers where's the peck of pickled peppers peter piper picked"
    total_utterances = 0
    transcription_set = set()

    transcription_lengths = []
    word_frequencies = {}
    for wav_file in glob.glob('/group/teaching/asr/labs/recordings/*.wav'):
        transcription = read_transcription(wav_file).split()
        
        if "".join(transcription) not in transcription_set:
            transcription_set.add("".join(transcription))

        transcription_lengths.append(len(transcription))
        
        for word in transcription:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

        total_utterances += 1

    fig = plt.figure(figsize=(10, 5))

    plt.boxplot(transcription_lengths, vert=0)
    plt.tick_params(axis='y', which='both', labelsize=25, left=False, right=False)
    plt.yticks([])
    plt.xlabel("Sentence Length", fontsize=20)
    plt.tight_layout()
    plt.savefig('boxplot.png', dpi=500)

    # Plot word distribution
    fig = plt.figure(figsize=(18, 9))
    plt.bar(word_frequencies.keys(), word_frequencies.values(), color='skyblue')
    plt.tick_params(axis='y', labelsize=25)
    plt.ylabel('Frequency Count', fontsize=30)
    plt.xticks(rotation=45, ha='right', fontsize=19)
    plt.savefig('barplot.png', dpi=500)

    print(f"Distinct Transcriptions (%): {len(transcription_set) / total_utterances}")


def compute_trigram_probs(dataset):
    trigram_counts = dict()
    bigram_counts = dict()
    unigram_counts = dict()

    words = "peter piper picked a peck of pickled peppers where's the peck of pickled peppers peter piper picked"
    words = list(set(words.split()))
    for word1 in words:
        for word2 in words:
            for word3 in words:
                trigram_counts[f"{word1}_{word2}_{word3}"] = 0

    for wav_file in dataset:
        transcription = read_transcription(wav_file).split(' ')

        for word1, word2, word3 in zip(transcription[:-2], transcription[1:-1], transcription[2:]):
            trigram_counts[f"{word1}_{word2}_{word3}"] += 1

        for word1, word2 in zip(transcription[:-1], transcription[1:]):
            bigram_counts[f"{word1}_{word2}"] = bigram_counts.get(f"{word1}_{word2}", 0) + 1
        
        for word in transcription:
            unigram_counts[word] = unigram_counts.get(word, 0) + 1

    for trigram in trigram_counts.keys():
        if trigram_counts[trigram] > 0:
            trigram_counts[trigram] /= bigram_counts[trigram[:trigram.rfind("_")]]
    return trigram_counts

def compute_final_probs(dataset):
    words = "peter piper picked a peck of pickled peppers where's the peck of pickled peppers peter piper picked"
    total_utterances = 0
    final_counts = {w: 0 for w in words.split(' ')}
    for wav_file in dataset:
        transcription = read_transcription(wav_file).split(' ')
        
        final_counts[transcription[-1]] += 1
        total_utterances += 1

    final_probs = {w: (float(final_counts[w]) / total_utterances) for w in words.split(' ')}
    return final_probs

def compute_unigram_probs(dataset):
    words = "peter piper picked a peck of pickled peppers where's the peck of pickled peppers peter piper picked"
    total_words = 0
    unigram_counts = {w: 0 for w in words.split(' ')}
    for wav_file in dataset:
        transcription = read_transcription(wav_file).split(' ')

        for word in transcription:
            total_words += 1
            unigram_counts[word] += 1

    unigram_probs = {w: (float(unigram_counts[w]) / total_words) for w in words.split()}
    return unigram_probs

def compute_bigram_probs(dataset):
    words = "peter piper picked a peck of pickled peppers where's the peck of pickled peppers peter piper picked"
    unigram_counts = {w: 0 for w in words.split()}
    bigram_counts = {}
    word_combs = []
    for word1 in words.split():
        for word2 in words.split():
            word_combs.append((word1, word2))
            bigram_counts[f"{word1}_{word2}"] = 0

    for wav_file in dataset:
        transcription = read_transcription(wav_file).split(' ')

        for word1 in transcription:
            unigram_counts[word1] += 1

        for word1, word2 in zip(transcription[:-1], transcription[1:]):
            bigram_counts[f"{word1}_{word2}"] += 1

    bigram_probs = {f"{word1}_{word2}": float(bigram_counts[f"{word1}_{word2}"] / unigram_counts[word1]) for word1, word2 in word_combs}

    return bigram_probs


def compute_perplexity(decoded, bigram_probs):
    total_prob = 0
    for word1, word2 in zip(decoded[:-1], decoded[1:]):
        if bigram_probs[f"{word1}_{word2}"] > 0:
            total_prob -= math.log(bigram_probs[f"{word1}_{word2}"])

    if total_prob == 0:
        return 0

    total_prob *= 1 / (len(decoded) + 1)

    return 2 ** total_prob

def read_transcription(wav_file):
    """
    Get the transcription corresponding to wav_file.
    """

    transcription_file = os.path.splitext(wav_file)[0] + '.txt'

    with open(transcription_file, 'r') as f:
        transcription = f.readline().strip()

    return transcription

def get_logs():
    log_directory = './new_logs/*.txt'

    best_model = None
    best_wer = 1e10
    for file in glob.glob(log_directory):
        with open(file, "r") as f:
            log_lines = f.readlines()
            wer = float(log_lines[-9].split(':')[1].strip())

            if wer < best_wer:
                best_wer = wer
                best_model = file.strip('./new_logs/')

    print(f"Best WER: {best_wer} | Best Model: {best_model}")
            

