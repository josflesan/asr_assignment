import os
import math
import glob
import matplotlib.pyplot as plt
from subprocess import check_call
from IPython.display import Image

def display_fst(f, filename="tmp.png"):
    f.draw('tmp.dot', portrait=True)
    check_call(['dot','-Tpng','-Gdpi=1600','tmp.dot','-o', filename])
    return Image(filename=filename)

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

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    plt.boxplot(transcription_lengths, vert=0)
    plt.title("Transcription Length Distribution (329 utts.)")
    plt.tick_params(axis='y', which='both', left=False, right=False)
    plt.xlabel("Sentence Length")
    plt.savefig('boxplot.png', dpi=300)

    # Plot word distribution
    fig = plt.figure(figsize=(8, 4))
    plt.bar(word_frequencies.keys(), word_frequencies.values(), color='skyblue')
    plt.ylabel('Frequency Count')
    plt.title('Word Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.savefig('barplot.png', dpi=300)

    print(f"Distinct Transcriptions (%): {len(transcription_set) / total_utterances}")



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
        return 1e10

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
