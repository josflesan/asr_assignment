import openfst_python as fst
import math
from helper_functions import parse_lexicon, generate_symbol_tables

lex = parse_lexicon('lexicon.txt')
word_table, phone_table, state_table = generate_symbol_tables(lex)

# MANUALLY BUILDS THE WFST -- gets same WER as others
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

        silence_final_state = generate_linear_silence_wfst(f, start_state, num_states, self_loop_prob)
        f.add_arc(silence_final_state, fst.Arc(0, 0, fst.Weight("log", 0.0), start_state))
        f.set_final(silence_final_state)

    elif use_sil:
        silence_final_state = generate_silence_wfst(f, start_state, 5, self_loop_prob)
        f.add_arc(silence_final_state, fst.Arc(0, 0, fst.Weight("log", 0.0), start_state))
        f.set_final(silence_final_state)

    if bigram_probs:
        for start_state, start_word in start_of_word_states:
            for end_state, end_word in end_of_word_states:
                if bigram_probs[f"{end_word}_{start_word}"] > 0:
                    f.add_arc(end_state, fst.Arc(0, 0, fst.Weight("log", -math.log(bigram_probs[f"{end_word}_{start_word}"])), start_state))
                else:
                    f.add_arc(end_state, fst.Arc(0, 0, fst.Weight("log", 1e10), start_state))

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
                # print(current_node, phone, edges[current_node].items())
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

def generate_L_wfst(lex):
    """ Express the lexicon in WFST form
    
    Args:
        lexicon (dict): lexicon to use, created from the parse_lexicon() function
    
    Returns:
        the constructed lexicon WFST
    
    """
    L = fst.Fst('log')
    
    # create a single start state
    start_state = L.add_state()
    L.set_start(start_state)
    L.set_final(start_state)
    
    root_state = L.add_state()
    L.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log", 0.0), root_state))
    for (word, pron) in lex.items():
        
        current_state = root_state
        for (i,phone) in enumerate(pron):
            next_state = L.add_state()
            next_output = 0 if i > 0 else word_table.find(word)
            next_weight = 0 if i > 0 else -math.log(1 / len(lex))
            
            if i == len(pron)-1:
                # add word output symbol on the final arc
                L.add_arc(current_state, fst.Arc(phone_table.find(phone), next_output, fst.Weight("log", next_weight), next_state))
            else:

                if i == 0:
                    L.add_arc(current_state, fst.Arc(phone_table.find(phone), next_output, fst.Weight("log", next_weight), next_state))
                else:
                    L.add_arc(current_state, fst.Arc(phone_table.find(phone), next_output, fst.Weight("log", next_weight), next_state))
            
            current_state = next_state
                          
        L.set_final(current_state)
        L.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", 0.0), root_state))
        
    L.set_input_symbols(phone_table)
    L.set_output_symbols(word_table)                      
    
    return L

def generate_G_wfst(wseq, unigram_probs=None):
    """ Generate a grammar WFST that accepts any sequence of words for words in a sentence.
        The bigrams not present in the sentence have a cost of 1, while those present have a cost of 0.
        Args:
            wseq (str): the sentence to use
        Returns:
            W (fst.Fst()): the grammar WFST """

    G = fst.Fst('log')
    start_state = G.add_state()
    G.set_start(start_state)

    prev_state = None
    word_start_states = dict()
    word_end_states = dict()

    bigrams = set(zip(wseq.split()[:-1], wseq.split()[1:]))
    words = set(wseq.split())
    for w in words:
        current_state = G.add_state()
        # word_start_states[w] = current_state

        weight = fst.Weight("log", -math.log(unigram_probs[w])) if unigram_probs else fst.Weight("log", -math.log(1 / len(words))) # if w == wseq.split()[0] else fst.Weight("log", 0.0)
        G.add_arc(start_state, fst.Arc(word_table.find("<eps>"), word_table.find(w), weight, current_state))

        prev_state = current_state
        current_state = G.add_state()

        G.add_arc(prev_state, fst.Arc(word_table.find(w), word_table.find(w), fst.Weight("log", 0.0), current_state))
        G.set_final(current_state)
        word_end_states[w] = current_state

        G.add_arc(current_state, fst.Arc(word_table.find("<eps>"), word_table.find("<eps>"), fst.Weight("log", 0.0), start_state))

        for w2, w2_state in word_start_states.items():
             weight = fst.Weight("log", -math.log(unigram_probs[w2])) if unigram_probs else fst.Weight("log", -math.log(1 / len(words))) # if (w, w2) in bigrams else fst.Weight('log', 1.0)
             G.add_arc(current_state, fst.Arc(word_table.find("<eps>"), word_table.find("<eps>"), weight, w2_state))

             if w != w2:
                 weight = fst.Weight("log", -math.log(unigram_probs[w])) if unigram_probs else fst.Weight("log", -math.log(1 / len(words))) # if (w2, w) in bigrams else fst.Weight('log', 1.0)
                 G.add_arc(
                     word_end_states[w2],
                     fst.Arc(
                         word_table.find("<eps>"),
                         word_table.find("<eps>"),
                         weight,
                         word_start_states[w]
                     )
                 )

    G.set_final(start_state)

    G.set_input_symbols(word_table)
    G.set_output_symbols(word_table)

    return G

def generate_H_wfst(self_loop_prob):
    H = fst.Fst('log')

    # create a single start state
    start_state = H.add_state()
    H.set_start(start_state)
    H.set_final(start_state)

    for _, phone in phone_table:
        if phone == "<eps>":
            continue

        current_state = H.add_state()
        H.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log", 0.0), current_state))
        current_state = generate_phone_wfst(H, current_state, phone, 3, self_loop_prob)

        H.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", -math.log(1 - self_loop_prob)), start_state))

    # current_state = H.add_state()
    # H.add_arc(start_state, fst.Arc(0, 0, fst.Weight("log", -math.log(1 / (phone_table.num_symbols() + 1))), current_state))
    # current_state = generate_silence_wfst(H, current_state, 5, self_loop_prob)

    # H.add_arc(current_state, fst.Arc(0, 0, fst.Weight("log", 0.0), start_state))
    # H.set_final(current_state)

    H.set_input_symbols(state_table)
    H.set_output_symbols(phone_table)

    return H
