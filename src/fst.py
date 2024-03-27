import openfst_python as fst
import math
from helper_functions import parse_lexicon, generate_symbol_tables

lex = parse_lexicon('lexicon.txt')
word_table, phone_table, state_table = generate_symbol_tables(lex)

def generate_sequence_wfst(word_seq, n=3, self_loop_prob=0.1, unigram_probs=None, final_probs=None, use_sil=None, bigram_probs=None):
    f = fst.Fst('log')
    start_state = f.add_state()
    f.set_start(start_state)

    # Start and end of word states
    start_of_word_states = []
    end_of_word_states = []

    words = set(word_seq.split())
    num_words = len(words)
    for word in words:
        current_state = f.add_state()
        start_of_word_states.append((current_state, word))

        word_prob = -math.log(1 / num_words)
        if unigram_probs:
            word_prob = -math.log(unigram_probs[word])

        f.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log", word_prob), current_state))

        num_phones = len(lex[word])
        for i, phone in enumerate(lex[word]):
            for j in range(1, n+1):
                in_label = state_table.find(f"{phone}_{j}")
                out_label = 0 if i+1 < num_phones or j != n else word_table.find(word)
                next_state = f.add_state()

                f.add_arc(current_state, fst.Arc(in_label, 0, fst.Weight("log", -math.log(self_loop_prob)), current_state))
                f.add_arc(current_state, fst.Arc(in_label, out_label, fst.Weight("log", -math.log(1-self_loop_prob)), next_state))

                current_state = next_state

        if final_probs:
            final_state = f.add_state()

            final_weight = 10.0 if final_probs[word] == 0 else -math.log(final_probs[word])
            f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", final_weight), final_state))

            if not bigram_probs:
                f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", 0.0), start_state))

            f.set_final(final_state)

            end_of_word_states.append((current_state, word))

        else:
            if not bigram_probs:
                f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", 0.0), start_state))
            f.set_final(current_state)

            end_of_word_states.append((current_state, word))

    if use_sil and use_sil.split('-')[0] == 'linear':
        num_states = int(use_sil.split('-')[1])
    
        for end_state, end_word in end_of_word_states:
            silence_start_state = f.add_state()
            silence_final_state = generate_linear_silence_wfst(f, silence_start_state, num_states, self_loop_prob)
            f.add_arc(end_state, fst.Arc(0, 0, fst.Weight("log", 0.0), silence_start_state))
            f.add_arc(silence_final_state, fst.Arc(0, 0, fst.Weight("log", 0.0), end_state))

        for initial_state, start_word in start_of_word_states:
            silence_start_state = f.add_state()
            silence_final_state = generate_linear_silence_wfst(f, silence_start_state, num_states, self_loop_prob)
            f.add_arc(initial_state, fst.Arc(0, 0, fst.Weight("log", 0.0), silence_start_state))
            f.add_arc(silence_final_state, fst.Arc(0, 0, fst.Weight("log", 0.0), initial_state))

    elif use_sil:

        for end_state, end_word in end_of_word_states:
            silence_start_state = f.add_state()
            silence_final_state = generate_silence_wfst(f, silence_start_state, 5, self_loop_prob)
            f.add_arc(end_state, fst.Arc(0, 0, fst.Weight("log", 0.0), silence_start_state))
            f.add_arc(silence_final_state, fst.Arc(0, 0, fst.Weight("log", 0.0), end_state))

        for initial_state, start_word in start_of_word_states:
            silence_start_state = f.add_state()
            silence_final_state = generate_silence_wfst(f, silence_start_state, 5, self_loop_prob)
            f.add_arc(initial_state, fst.Arc(0, 0, fst.Weight("log", 0.0), silence_start_state))
            f.add_arc(silence_final_state, fst.Arc(0, 0, fst.Weight("log", 0.0), initial_state))

    if bigram_probs:
        for initial_state, start_word in start_of_word_states:
            for end_state, end_word in end_of_word_states:
                if bigram_probs[f"{end_word}_{start_word}"] > 0:
                    f.add_arc(end_state, fst.Arc(0, 0, fst.Weight("log", -math.log(bigram_probs[f"{end_word}_{start_word}"])), initial_state))
                else:
                    f.add_arc(end_state, fst.Arc(0, 0, fst.Weight("log", 1e10), initial_state))

    f.set_input_symbols(state_table)
    f.set_output_symbols(word_table)
    return f

def generate_sequence_trigram_wfst(word_seq, n=3, self_loop_prob=0.1, unigram_probs=None, final_probs=None, use_sil=False, trigram_probs=None):
    f = fst.Fst('log')
    start_state = f.add_state()
    f.set_start(start_state)

    words = list(set(word_seq.split()))
    num_words = len(words)

    # Start and end of word states
    start_of_bigram_states = {word: dict() for word in words}
    end_of_bigram_states = {word: dict() for word in words}

    for word1 in words:
        for word2 in words:
            current_state = f.add_state()
            start_of_bigram_states[word1][word2] = current_state

            word_prob = -math.log(1 / num_words)
            if unigram_probs:
                word_prob = -math.log(unigram_probs[word2])

            f.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log", word_prob), current_state))

            num_phones = len(lex[word2])
            for i, phone in enumerate(lex[word2]):
                for j in range(1, n+1):
                    in_label = state_table.find(f"{phone}_{j}")
                    out_label = 0 if i+1 < num_phones or j != n else word_table.find(word2)
                    next_state = f.add_state()

                    f.add_arc(current_state, fst.Arc(in_label, 0, fst.Weight("log", -math.log(self_loop_prob)), current_state))
                    f.add_arc(current_state, fst.Arc(in_label, out_label, fst.Weight("log", -math.log(1-self_loop_prob)), next_state))

                    current_state = next_state

            if final_probs:
                final_state = f.add_state()

                final_weight = 10.0 if final_probs[word2] == 0 else -math.log(final_probs[word2])
                f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", final_weight), final_state))

                if not trigram_probs:
                    f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", 0.0), start_state))

                f.set_final(final_state)
                end_of_bigram_states[word1][word2] = current_state

            else:
                if not trigram_probs:
                    f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", 0.0), start_state))
                f.set_final(current_state)

                end_of_bigram_states[word1][word2] = current_state

    if use_sil:
        for start_word in end_of_bigram_states.keys():
            for middle_word in end_of_bigram_states[start_word].keys():
                silence_start_state = f.add_state()
                silence_final_state = generate_silence_wfst(f, silence_start_state, 5, self_loop_prob)
                bigram_end_state = end_of_bigram_states[start_word][middle_word]
                f.add_arc(bigram_end_state, fst.Arc(0, 0, fst.Weight("log", 0.0), silence_start_state))
                f.add_arc(silence_final_state, fst.Arc(0, 0, fst.Weight("log", 0.0), bigram_end_state))
        for start_word in start_of_bigram_states.keys():
            for middle_word in start_of_bigram_states[start_word].keys():
                silence_start_state = f.add_state()
                silence_final_state = generate_silence_wfst(f, silence_start_state, 5, self_loop_prob)
                bigram_start_state = end_of_bigram_states[start_word][middle_word]
                f.add_arc(bigram_start_state, fst.Arc(0, 0, fst.Weight("log", 0.0), silence_start_state))
                f.add_arc(silence_final_state, fst.Arc(0, 0, fst.Weight("log", 0.0), bigram_start_state))


    if trigram_probs:
        for start_word in end_of_bigram_states.keys():
            for middle_word in end_of_bigram_states[start_word].keys():
                for final_word in start_of_bigram_states[middle_word].keys():
                    end_state = end_of_bigram_states[start_word][middle_word]
                    initial_state = start_of_bigram_states[middle_word][final_word]
                    trigram_prob = trigram_probs[f"{start_word}_{middle_word}_{final_word}"]
                    if trigram_prob > 0:
                        f.add_arc(end_state, fst.Arc(0, 0, fst.Weight("log", -math.log(trigram_prob)), initial_state))
                    else:
                        f.add_arc(end_state, fst.Arc(0, 0, fst.Weight("log", 1e10), initial_state))

    f.set_input_symbols(state_table)
    f.set_output_symbols(word_table)
    return f

def generate_sequence_alt_pronunciations_wfst(word_seq, n=3, self_loop_prob=0.1, unigram_probs=None, final_probs=None, use_sil=False, use_pronounciations=False):
    f = fst.Fst('log')
    start_state = f.add_state()
    f.set_start(start_state)

    words = list(set(word_seq.split()))
    num_words = len(words)

    for word in words:
        if not use_pronounciations:
            lex[word] = [lex[word][0]]
        for pronunciation in lex[word]:
            current_state = f.add_state()

            weight = num_words
            if unigram_probs:
                weight = unigram_probs[word]
            word_prob = -math.log(1 / (weight * len(lex[word])))
            f.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log", word_prob), current_state))
            
            num_phones = len(pronunciation)
            for i, phone in enumerate(pronunciation):
                for j in range(1, n+1):
                    in_label = state_table.find(f"{phone}_{j}")
                    out_label = 0 if i+1 < num_phones or j != n else word_table.find(word)
                    next_state = f.add_state()

                    f.add_arc(current_state, fst.Arc(in_label, 0, fst.Weight("log", -math.log(self_loop_prob)), current_state))
                    f.add_arc(current_state, fst.Arc(in_label, out_label, fst.Weight("log", -math.log(1-self_loop_prob)), next_state))

                    current_state = next_state

        if final_probs:
            final_state = f.add_state()

            final_weight = 10.0 if final_probs[word] == 0 else -math.log(final_probs[word])
            f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", final_weight), final_state))
            f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", 0.0), start_state))
            f.set_final(final_state)
        else:
            f.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", 0.0), start_state))
            f.set_final(current_state)

    if use_sil:
        silence_start_state = f.add_state()
        f.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log", -math.log(1 / num_words)), silence_start_state))
        silence_final_state = generate_silence_wfst(f, silence_start_state, 5, self_loop_prob)
        f.add_arc(silence_final_state, fst.Arc(0, 0, fst.Weight("log", 0.0), start_state))
        f.set_final(silence_final_state)

    f.set_input_symbols(state_table)
    f.set_output_symbols(word_table)
    return f

def generate_tree_wfst(word_seq, n=3, self_loop_prob=0.1):
    words = set(word_seq.split())
    nodes, edges, sizes = [(None, None)], [dict()], [0]

    for word in words:
        current_node = 0
        phones = lex[word]
        for phone in phones:
            if phone not in edges[current_node]:
                edges[current_node][phone] = len(nodes)
                nodes.append((phone, 0)) # input phone, output word
                edges.append(dict())
                sizes.append(0)
            current_node = edges[current_node][phone]
        
        nodes[-1] = (nodes[-1][0], word_table.find(word)) # output word at the last phone
    
    f = fst.Fst('log')
    start_state = f.add_state()
    f.set_start(start_state)
    f.set_final(start_state)

    def fill_sizes(node):
        if len(edges[node]) == 0:
            sizes[node] = 1
        else:
            for next_node in edges[node].values():
                fill_sizes(next_node)
                sizes[node] += sizes[next_node]
    
    def generate_tree(node, current_state):
        phone = nodes[node][0]        
        for i in range(1, n+1):
            in_label = state_table.find(f"{phone}_{i}")
            f.add_arc(current_state, fst.Arc(in_label, 0, fst.Weight("log", -math.log(self_loop_prob)), current_state))

            if i != n:
                next_state = f.add_state()
                f.add_arc(current_state, fst.Arc(in_label, 0, fst.Weight("log", -math.log(1-self_loop_prob)), next_state))
                current_state = next_state

        final_prob = -math.log(1-self_loop_prob)
        if len(edges[node]) == 0:
            f.add_arc(current_state, fst.Arc(in_label, nodes[node][1], fst.Weight("log", final_prob), start_state))
            
        for next_node in edges[node].values():
            transition_prob = -math.log(sizes[next_node]/sizes[node]) + final_prob
            
            next_state = f.add_state()
            f.add_arc(current_state, fst.Arc(in_label, 0, fst.Weight("log", transition_prob), next_state))
            generate_tree(next_node, next_state)
    
    fill_sizes(0)
    for next_node in edges[0].values():
        transition_prob = -math.log(sizes[next_node]/sizes[0])
        
        next_state = f.add_state()
        f.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log", transition_prob), next_state))
        generate_tree(next_node, next_state)
    
    f.set_input_symbols(state_table)
    f.set_output_symbols(word_table)

    return f

def generate_linear_silence_wfst(f, start_state, n, self_loop_prob=0.1):
    current_state = start_state
    for i in range(1, n+1):
        in_label = state_table.find('sil_{}'.format(i))
        sl_weight = fst.Weight('log', -math.log(self_loop_prob))
        f.add_arc(current_state, fst.Arc(in_label, 0, sl_weight, current_state))

        next_state = f.add_state()
        next_weight = fst.Weight('log', -math.log(1 - self_loop_prob))
        f.add_arc(current_state, fst.Arc(in_label, 0, next_weight, next_state))

        current_state = next_state

    return current_state

def generate_silence_wfst(f, start_state, n, self_loop_prob=0.1):
    current_state = start_state
    states = [start_state]

    for i in range(1, n+1):
        in_label = state_table.find('sil_{}'.format(i))

        sl_weight = fst.Weight('log', -math.log(self_loop_prob))
        f.add_arc(current_state, fst.Arc(in_label, 0, sl_weight, current_state))

        next_state = f.add_state()
        states.append(next_state)

        if i == 2 or i == 4:
            next_weight = fst.Weight('log', -math.log((1 - self_loop_prob) / 3))
        elif i == 3:
            next_weight = fst.Weight('log', -math.log((1 - self_loop_prob) / 2))
        else:
            next_weight = fst.Weight('log', -math.log(1 - self_loop_prob))

        f.add_arc(current_state, fst.Arc(in_label, 0, next_weight, next_state))

        current_state = next_state

    f.set_final(current_state)
    final_state = current_state

    # Create ergodic loops
    for i in range(2, 5):
        current_state = states[i - 1]
        for j in range(2, 5):
            if j == i or j == i+1: 
                continue
            next_state = states[j-1]

            in_label = state_table.find('sil_{}'.format(i))
            if i == 2 or i == 4:
                weight = fst.Weight('log', (1 - self_loop_prob) / 3)
            else:
                weight = fst.Weight('log', (1 - self_loop_prob) / 2)

            f.add_arc(current_state, fst.Arc(in_label, 0, weight, next_state))

    return final_state

def generate_phone_wfst(f, start_state, phone, n, self_loop_prob=0.1):
    """
    Generate a WFST representing an n-state left-to-right phone HMM.

    Args:
        f (fst.Fst()): an FST object, assumed to exist already
        start_state (int): the index of the first state, assumed to exist already
        phone (str): the phone label
        n (int): number of states of the HMM excluding start and end
        self_loop_prob (float): self loop probability

    Returns:
        the final state of the FST
    """

    current_state = start_state

    for i in range(1, n+1):

        in_label = state_table.find('{}_{}'.format(phone, i))

        sl_weight = fst.Weight('log', -math.log(self_loop_prob))  # weight for self loop
        # self-loop back to current state
        f.add_arc(current_state, fst.Arc(in_label, 0, sl_weight, current_state))

        # transition to next state

        # we want to output the phone label on the final state
        # note: if outputting words instead this code should be modified
        if i == n:
            out_label = phone_table.find(phone)
        else:
            out_label = 0   # output empty <eps> label

        next_state = f.add_state()
        next_weight = fst.Weight('log', -math.log(1 - self_loop_prob))  # weight for next state
        f.add_arc(current_state, fst.Arc(in_label, out_label, next_weight, next_state))

        current_state = next_state
    return current_state
