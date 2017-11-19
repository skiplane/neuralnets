
from collections import Counter
import numpy as np
from helpers import convert_dates_to_dummy_day, remove_punctuation
import difflib # for longest pattern match
import sys, yaml
from tensorflow.python.lib.io import file_io
import gensim

stop_words = ['book', 'a', 'an', 'to', 'abeam', 'aboard', 'about', 'above', 'abreast', 'abroad', 'absent', 'across',
                  'adjacent', 'after', 'against',
                  'again', 'according', 'adjacent', 'aside', 'astern', 'due', 'except', 'far', 'against', 'again',
                  'amid', 'belong', 'beyond',
                  'apart', 'as', 'gain', 'along', 'alongst', 'alongside', 'amid', 'amidst', 'mid', 'midst', 'among',
                  'amongst', 'apropos', 'apud',
                  'around', 'round', 'as', 'astride', 'at', '@', 'atop', 'ontop', 'bar', 'before', 'afore', 'tofore',
                  'behind', 'ahind', 'below',
                  'ablow', 'allow', 'beneath', 'neath', 'back', 'beside', 'because', 'besides', 'between', 'beyond',
                  'but', 'by', 'chez', 'circa', 'come', 'close',
                  'dehors', 'despite', 'spite', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into',
                  'less', 'like', 'minus', 'instead',
                  'near', 'nearer', 'anear', 'notwithstanding', 'next', 'of', 'off', 'on', 'onto', 'opposite', 'out',
                  'outen', 'outside', 'over',
                  'pace', 'past', 'per', 'post', 'pre', 'pro', 'prior', 'pursuant', 'qua', 're', 'sans', 'save', 'sauf',
                  'short', 'since', 'sithence', 'than', 'left',
                  'through', 'thru', 'throughout', 'thruout', 'to', 'toward', 'towards', 'under', 'underneath',
                  'unlike', 'until', 'up', 'upon', 'rather',
                  'upside', 'versus', 'via', 'vice', 'vis-a-vis', 'with', 'within', 'without', 'worth', 'for', 'per',
                  'regards',
                  'opposite', 'out', 'outside', 'owing', 'regardless', 'right', 'subsequent', 'such', 'thanks', 'upto',
                  'up', 'opposed', 'soon',
                  'well', 'behest', 'means', 'virtue', 'sake', 'accordance', 'addition', 'case', 'front', 'lieu',
                  'order', 'place', 'point', 'spite',
                  'account', 'behalf', 'top', 'regard', 'respect', 'until', 'uptill', 'atop', 'forth', 'next', 'nigh',
                  'unto', 'ago', 'apart', 'aside',
                  'away', 'hence', 'notwithstanding', 'not', 'no', 'yes', 'on', 'through', 'withal', 'opposed',
                  'middle', 'dint', 'way', 'sake',
                  'advance', 'face', 'favor', 'lieu', 'hereabout', 'hereabouts', 'hereafter', 'hereat', 'hereby',
                  'herefor', 'herefore', 'herefrom',
                  'herein', 'hereinafter', 'hereinbefore', 'hereinto', 'hereof', 'hereon', 'hereto', 'heretofore',
                  'hereunder', 'hereunto', 'hereupon',
                  'herewith', 'herewithal', 'herewithin', 'thereabout', 'thereabouts', 'thereacross', 'thereafter',
                  'thereagainst', 'therearound',
                  'thereat', 'therebeyond', 'thereby', 'therefor', 'therefore', 'therefrom', 'therein', 'thereinafter',
                  'thereinbefore', 'thereinto',
                  'thereof', 'thereon', 'thereover', 'therethrough', 'thereto', 'thereupon', 'thereunder',
                  'therewithal', 'therewithin',
                  'whereabout', 'whereabouts', 'whereafter', 'whereagainst', 'whereas', 'whereat', 'whereby',
                  'wherefor', 'wherefore', 'wherefrom',
                  'wherein', 'whereinto', 'whereof', 'whereon', 'whereover', 'whereunder', 'whereunto', 'whereupon',
                  'wherever', 'wherewith', 'wherewithal',
                  'wherewithin', 'wherewithout']

'''
Cloud compatible yaml read
'''
def load_yaml(filename):
    dat = file_io.read_file_to_string(filename)
    return yaml.load(dat)

'''
Load file with training data of lines containing sentences and labels separated by comma.
'''
def load_data(path, to_lower = True):
    with open(path,'r') as f:
        lines = f.readlines()
    queries = []
    for l in lines:
        if to_lower:
            l = l.lower()
        processed = l.strip('\n').split(',')
        queries.append( (processed[len(processed)-2]).split() ) # A list of lists. each element is a list of the words of a query
    values = []
    for l in lines:
        # print "line: ", l
        if to_lower:
            l = l.lower()
        processed = l.strip('\n').split(',')
        value = processed[len(processed)-1]
        value = value.strip()
        values.append(value)  # A list. Each element is an intent
    return queries, values

'''
For training: X, Y = convert_to_numbers(trainX = trainX, trainY = trainY, word2vec_model = word2vec_model )
For prediction: X, _ =  convert_to_numbers(trainX = trainX, word2vec_model = word2vec_model)
'''
def convert_to_numbers(word2vec_model, sequence_len, num_labels, wordvec_size, func, trainX, trainY = None):

    batch_size = len(trainX)
    X = np.zeros((batch_size, sequence_len, wordvec_size), dtype='float32')
    Y = np.zeros((batch_size, num_labels), dtype='float32')

    for batch in xrange(batch_size):
        sentence = trainX[batch]
        for idx in xrange(min(sequence_len,len(sentence))):
            w = sentence[idx]
            if w in word2vec_model:
                embed = word2vec_model[w]
            else:
                embed = np.zeros(wordvec_size, dtype='float32')
            X[batch][idx] = embed[:]

        if trainY:
            label = trainY[batch]
            Y[batch][func(label)] = 1.

    return X, Y

def print_layer_parameters(model, layers, outputfile=None):
    idx = 1
    if outputfile:
        f = open(outputfile, 'a+')
    for layer in layers:
        try:
            if outputfile:
                f.write("\nLayer: %d\n" % idx)
                f.write("Weights:\n")
                f.write("%s" % model.get_weights(layer.W))
                f.write("Biases:")
                f.write("%s" % model.get_weights(layer.b))
                f.write("\n------------------")
            else:
                print("\nLayer: %d\n" % idx)
                print "Weights:"
                print model.get_weights(layer.W)
                print "Biases:"
                print model.get_weights(layer.b)
                print "\n------------------"
            idx += 1
        except:
            pass
    if outputfile:
        f.close()

'''
The usual cleaning of sentences before sending for conversion to number arrays
'''
def usual_clean(sentence = '', punctuations_to_remove = '.!?"~;,-/()'):
    sentence = convert_dates_to_dummy_day(sentence)
    sentence = remove_punctuation(sentence=sentence, punctuations= punctuations_to_remove)
    sentence = sentence.strip(' ')
    return sentence

'''
Get degree of similarity in spelling between two words
'''
def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2))
    return s1[pos_a:pos_a+size]

'''
Return a list of sentences that are perturbations obtained from a single sentence VIA WORD SUBSTITUTIONS
'''
def word_substitutions(words_in, word_model, max_similars=10):
    '''
    :param words_in: list of words
    '''
    words = []
    for word in words_in:
        words.append(unicode(word, 'utf-8'))

    sents = []
    for i in xrange(len(words)):
        word = words[i]
        if (word in word_model) and (word not in stop_words):
            res = word_model.most_similar(word, topn=50)
            similars = []
            for s,_ in res:
                if len(similars) == max_similars:
                    break
                s = s.lower()
                too_similar_to_prev = False
                if len(similars) > 0:
                    for s2 in similars:
                        too_similar_to_prev = len(s) > 4 and  (len(s) - len(get_overlap(s,s2)) < 3)
                        if too_similar_to_prev:
                            break
                if not too_similar_to_prev:
                    similars.append(s)

            for level, sim in enumerate(similars):
                sim_word = sim.lower()
                if ('_' not in sim_word) and ('.' not in sim_word):
                    sent = " ".join(words[:i]) + " " + sim_word + " " + " ".join(words[i+1:])
                    sents.append( (sent, level+1))
    return sents # list of tuples (sent, level)

'''
Generate confidence score from output array using kurtosis
Y: input array, ex: [0.34, 0.5, 0.2, 0.001, 0.0034] --> which indicates hotel
'''
def kurtosis(x):
    avg = np.average(x)
    numer = 0.0
    den = 0.0
    for val in x:
        numer += (val - avg) ** 4
        den += (val - avg) ** 2
    den = den ** 2
    kurt = len(x) * (numer / den)
    return kurt

'''
Confidence score based on predictions on parts of the query
'''
def evaluate_on_windows(window_sz, query, orig_ans, neural_model, numbers_to_intents_func,
                        sequence_len, num_labels, wordvec_size):
    win_queries = []

    eff_win_sz = min(window_sz, len(query))
    for win_sz in xrange(1, eff_win_sz):
        for st in xrange(len(query) - eff_win_sz + 1):
            win_queries.append(query[st: st + eff_win_sz])

    # Run predictions on the perturbed queries
    X_new, _ = convert_to_numbers(trainX=win_queries, word2vec_model=word2vec_model, sequence_len=sequence_len,
                                  num_labels=num_labels, wordvec_size=wordvec_size)
    Y_new = neural_model.predict(X_new)

    win_answers = []
    for y in Y_new:
        win_answers.append(numbers_to_intents_func(y)[0])

    win_q = []
    for q in win_queries:
        win_q.append(" ".join(q))

    dist = Counter(win_answers)
    dist.pop('none', None)

    if dist:
        most_common = dist.most_common(1)[0][0]
    else:
        most_common = 'none'

    common_equals_orig_ans =  (most_common == orig_ans)
    num_answers = len(dist)

    return common_equals_orig_ans, num_answers

def combine_probabilities(P1_prime, P2_prime):
    val = 0.01
    if P1_prime > 0.5 and P2_prime > 0.5:
        val += P1_prime + ( (1-P1_prime) * (P2_prime-0.5)/P2_prime)
    elif P1_prime > 0.5 and P2_prime <= 0.5 and (abs(P1_prime-0.5) > abs(0.5-P2_prime)):
        val += P1_prime - (1-P1_prime)*(0.5-P2_prime)/0.5
    elif P1_prime > 0.5 and P2_prime <= 0.5 and (abs(0.5-P2_prime) >= abs(P1_prime-0.5)):
        val +=  P2_prime + (0.5-P2_prime)*(P1_prime-0.5)/0.5
    elif P1_prime <= 0.5:
        val += P2_prime - P2_prime * (0.5 - P1_prime)/P1_prime
    else:
        print "ERROR: unknown case of P1_prime and P2_prime"
        exit(1)

    if val >= 0.8:
        val = 0.95
    elif (val >= 0.2 and val < 0.8):
        val += 0.5
    elif (val >= 0.02 and val <0.2):
        val += 0.4
    elif (val >= 0.005 and val < 0.02):
        val += 0.3
    elif (val < 0.005):
        val += 0.1

    return val

'''
A way to compute confidence score using the method of shifting windows in models that classify with
layers of convolutional nets - Note. Convolutional nets look out for space invariant patterns
'''
def confidence_score(neural_model, method = 'shifting_windows', Y = None, orig_ans = None, query = None, window_sz=3, file_handle = None):
    P1_map = {0: 0.90, 1: 0.38, 2: 0.27, 3: 0.25, 4: 0.01}
    if method=='shifting_windows':
        common_equals_orig_ans, num_answers = evaluate_on_windows(window_sz=window_sz, query=query, orig_ans=orig_ans, neural_model=neural_model)
        text = "common_equals_orig_ans: "+  str(common_equals_orig_ans) + " num_answers: " + str(num_answers) + "\n"
        file_handle.write(text)

        if common_equals_orig_ans:
            P2 = 0.44
        else:
            P2 = 0.17
        if num_answers in P1_map:
            P1 = P1_map[num_answers]
        else:
            P1 = 0.01

        P1_prime, P2_prime = max(P1, P2), min(P1, P2)
        conf_score = combine_probabilities(P1_prime=P1_prime, P2_prime=P2_prime)
        print "Confidence score: ", conf_score
        text = "P1: "+str(P1)+" P2: "+ str(P2) +" P1_prime: "+ str(P1_prime) +" P2_prime: "+ str(P2_prime) +" score: "+ str(conf_score) +"\n"
        file_handle.write(text)
        return conf_score

    elif method=='kurtosis' and Y:
        # use peakedness to estimate confidence
        return kurtosis(Y)

    else:
        print "ERROR: Please specify correct method to use for evaluating confidence score"
        sys.exit(1)

if __name__=='__main__':
    saved_wordvec_model = '~/work/models/GoogleNews-vectors-negative300.bin'
    word2vec_model = gensim.models.Word2Vec.load_word2vec_format(saved_wordvec_model, binary=True)

    sentence = "Can you please book me a table in a restaurant in San Francisco?"
    sentence = usual_clean(sentence=sentence, punctuations_to_remove='.!?"~;,-/()')
    print "After cleaning: ", sentence
    sents = word_substitutions(sentence=sentence, word_model=word2vec_model, max_similars=20)
    print sents

    

