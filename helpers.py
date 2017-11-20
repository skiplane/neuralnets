
import pandas as pd
import re

'''
Dump columns from excel w/o processing in a csv file
'''
def dump_queries_from_excel(inputfile = '', outputfile = '', sheetnames = None, columns = None,
                            append = False, process_queries=False, punctuations_to_remove = None):
    mode = 'a' if append else 'w'
    with open(outputfile, mode) as out:
        if inputfile.endswith(".xlsx") or inputfile.endswith(".xls"):
            with pd.ExcelFile(inputfile) as xls:
                if not sheetnames:
                    sheetnames = xls.sheet_names
                for s in sheetnames:
                    df = pd.read_excel(xls, sheetname=s, na_values=['NA'])
                    outcols = [] # all inputs into a list
                    for col in columns:
                        outcol = []
                        for l in df[col]:
                            if isinstance(l, unicode) or isinstance(l, str):
                                if process_queries:
                                    l = convert_dates_to_dummy_day(l)
                                    l = remove_punctuation(sentence=l, punctuations=punctuations_to_remove)
                                if punctuations_to_remove:
                                    l = remove_punctuation(sentence=l, punctuations=punctuations_to_remove)
                                outcol.append(l)
                        outcols.append(outcol)
                    if len(outcols)==0:
                        print "No columns got extracted"
                        return
                    for i in xrange(len(outcols[0])):
                        for j in xrange(len(outcols)):
                            line = [l[i] for l in outcols]
                            line = ', '.join(line)
                            for c in line:
                                try:
                                    out.write(c)
                                except UnicodeEncodeError as e:
                                    print "1.a Warning.", e.message, "Bad unicode character here: ", c
                            out.write(u'\n')

'''
Read from excel file of particular format and generate an intermediate file with all queries

Excel file format: Column 'Query' has all queries with definite intent. Other cols 'Query.1', 'Query.2', etc may or may not have definite intent
'''
def process_queries_from_excel(inputfile = '', outputfile = '', sheetnames = None, columns_with_intent = None,
                 columns_without_intent = None, intent = '', append = False,
                 process_queries=False,
                 punctuations_to_remove = None):
    mode = 'a' if append else 'w'
    with open(outputfile, mode) as out:
        if inputfile.endswith(".xlsx") or inputfile.endswith(".xls"):
            with pd.ExcelFile(inputfile) as xls:
                for s in sheetnames:
                    df = pd.read_excel(xls, sheetname=s, na_values=['NA'])
                    for col in columns_with_intent:
                        for l in df[col]:
                            if isinstance(l,unicode) or isinstance(l, str):
                                try:
                                    if process_queries:
                                        l = convert_dates_to_dummy_day(l)
                                        l = remove_punctuation(sentence=l, punctuations=punctuations_to_remove)
                                    out.write(l + u',' + intent + u'\n')
                                except UnicodeEncodeError as e:
                                    print "1.a Warning.", e.message,"Bad unicode character here: ", l
                                except Exception as e:
                                    print "1.b Warning: Bad sentence: ", l," ",e.message
                    for col in columns_without_intent:
                        for l in df[col]:
                            if isinstance(l, unicode) or isinstance(l, str):
                                try:
                                    if process_queries:
                                        l = convert_dates_to_dummy_day(l)
                                        l = remove_punctuation(sentence=l, punctuations=punctuations_to_remove)
                                    out.write(l + u',' + 'none' + u'\n')
                                except UnicodeEncodeError as e:
                                    print "2.a Warning.", e.message,"Bad unicode character here: ", l
                                except Exception as e:
                                    print "2.b Warning: Bad sentence: ", l, " ", e.message


def remove_punctuation(sentence = '', punctuations = ''):
    for p in punctuations:
        sentence = sentence.replace(p,' ')
    sentence = sentence.replace('  ',' ')
    sentence = sentence.strip(' ')
    return sentence

def convert_dates_to_dummy_day(sent = ''):
    sent =  re.sub('\d+([ */-]*\d)*',' 1 ', sent)
    return re.sub('(\d[ */-]*\d?)+','1 ', sent)

def check_word2vec_keys(textfile='', outfile = '', model=None):
    with open(outfile,'w') as o:
        with open(textfile,'r') as f:
            lines = f.readlines()
            for l in lines:
                for w in l.replace(',',' ').replace('.',' ').split():
                    if w not in model:
                        o.write(w + u'\n')
'''
For processing text data file
'''
def clean_annotated(infile='', outfile = '', punctuations_to_remove='', dummy_dates = True):
    with open(infile,'r') as f:
        with open(outfile, 'w') as fout:
            lines = f.readlines()
            for l in lines:
                if dummy_dates:
                    l = convert_dates_to_dummy_day(l)
                l = remove_punctuation(sentence=l, punctuations=punctuations_to_remove)
                l = l.strip(' ')
                fout.write(l)


def chop_sentences_to_length(input_file, output_file, max_num_words=50):
    out_file = open(output_file, 'w')
    with open(input_file) as in_file:
        lines = in_file.readlines()
        for l in lines:
            query = l.split(',')[0]
            intent = l.split(',')[1]
            words = query.split()
            out = words[:max_num_words]
            out_file.write(" ".join(out) + ',' + intent)
    out_file.close()

def split_to_training_and_validation(input_file, training_file, validation_file, line_range = (1,10000), val_num = 1000):
    tr_file = open(training_file, 'w')
    val_file = open(validation_file, 'w')
    val_ctr = 0
    line_ctr = 0

    with open(input_file) as in_file:
        lines = in_file.readlines()
        max_line = min(line_range[1],len(lines))
        for l in lines[line_range[0]:]:
            line_ctr += 1
            if line_ctr < max_line and val_ctr < val_num and line_ctr % 5 == 0:
                val_file.write(l)
                val_ctr += 1
            else:
                tr_file.write(l)

    val_file.close()
    tr_file.close()

def plot_length_of_queries(input_file):
    lengths = []

    with open(input_file) as in_file:
        lines = in_file.readlines()
        for l in lines:
            lengths.append(len(l.split()))
    return lengths

if __name__=='__main__':
    dump_queries_from_excel(inputfile='data/intent_convs.xlsx',
                            outputfile='data/intent_convs.csv',
                            columns=['Query','Intent'],
                            append=False, process_queries=False, punctuations_to_remove='.!?"~;,-/()')
