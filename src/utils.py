import os
import glob
from subprocess import check_call
from IPython.display import Image

def display_fst(f, filename="tmp.png"):
    f.draw('tmp.dot', portrait=True)
    check_call(['dot','-Tpng','-Gdpi=1600','tmp.dot','-o', filename])
    return Image(filename=filename)

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
            for word2 in transcription:
                unigram_counts[word1] += 1
                bigram_counts[f"{word1}_{word2}"] += 1

    bigram_probs = {f"{word1}_{word2}": float(bigram_counts[f"{word1}_{word2}"] / unigram_counts[word1]) for word1, word2 in word_combs}

    print(bigram_probs)
    return bigram_probs


def read_transcription(wav_file):
    """
    Get the transcription corresponding to wav_file.
    """

    transcription_file = os.path.splitext(wav_file)[0] + '.txt'

    with open(transcription_file, 'r') as f:
        transcription = f.readline().strip()

    return transcription
