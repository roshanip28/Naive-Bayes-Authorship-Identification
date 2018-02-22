#!/usr/bin/env python
from collections import defaultdict
from csv import DictReader, DictWriter

import nltk
import codecs
import sys
import string
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer

kTOKENIZER = TreebankWordTokenizer()

def morphy_stem(word):
    """
    Simple stemmer
    """
    stem = wn.morphy(word)
    if stem:
        return stem.lower()
    else:
        return word.lower()

class FeatureExtractor:
    def __init__(self):
        """
        You may want to add code here
        """
        None
    def vowel_count(self,text,g):
        count=0
        for i in text:
            if i in 'aeiou':
                count=count+1
        g["vowel_count"]=count
        return g

    def sentence_length(self,text,g):
        count=0
        text=text.split()
        for i in text:
            count+=1
        g["sentence_length"]=count
        return g


    def avg_word_length(self,text,g):
        a=0
        b=0
        text=text.split()

        for j in text:
            if j not in string.punctuation and not j.isdigit():
                b+=1
                a+=len(j)

        g["word_avg"]=(a/b)
        return g

    def capital_word(self,text,g):
        count=0
        text = text.split()
        for j in text:
            if j[0].isupper():
                count+=1
        g["capital_word"]=count
        return g

    def lower_text(self,text,g):
        count=0
        h=0
        for i in text:
            if i.islower():
                h+=1
        g["lower_text"] = h
        return g

    def word_uprcase(self,text,g):
        count=0
        text = text.split()
        for i in text:
            if i.isupper():
                count+=1
        g["word_uprcase"]=count
        return g

    def start_end(self,text,g):
        text=text.split()
        g[morphy_stem(text[0])+"X"]=1
        g[morphy_stem(text[len(text)-1]) + "y"] = 1
        return g

    def punc_count(self,text,g):
        for i in text:
            if i in string.punctuation:
                g["punc"]+=1
        return g

    def sen_case_type(self,text,g):
        text=text.translate(None,string.punctuation)
        text=text.split()
        for i in range(0,len(text)):
            if i!=0 and text[i][0].isupper() and text[i]!="I":
                print text[i]
                g["sentence_case"]+=1
        #print text.split()
        return g

    def consecutive_con(self,text,g):
        text = text.split()
        for i in range(0, len(text)):
            if text[i] not in 'aeiou':
                if text[i]==text[i-1]:
                    g["conse_cons"]+=1
        return g
    def colons_freq(self,text,g):
        for i in text:
            if i ==":":
                g["colons"] += 1
        return g
    def stem_vowels(self,text,g):
        text=text.split(" ")
        for i in text:
               for j in morphy_stem(i):
                   if j in 'aeiou':
                       g["stem_vowels"] += 1
        return g



    def features(self, text):
        d = defaultdict(int)
        for ii in text.split(" "):
            d[morphy_stem(ii)] += 1
        self.vowel_count(text,d)
        self.sentence_length(text,d)
        self.avg_word_length(text,d)
        self.lower_text(text,d)
        self.word_uprcase(text,d)
        self.punc_count(text,d)
        self.colons_freq(text,d)
        #self.consecutive_con(text,d)
        #self.sen_case_type(text,d)
        #self.stem_vowels(text,d)
        #self.capital_word(text,d)
        #self.start_end(text,d)
        return d




reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code if code.endswith("t") else code+"t") if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--trainfile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input train file")
    parser.add_argument("--testfile", "-t", nargs='?', type=argparse.FileType('r'), default=None, help="input test file")
    parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this fraction of total')
    args = parser.parse_args()
    trainfile = prepfile(args.trainfile, 'r')
    if args.testfile is not None:
        testfile = prepfile(args.testfile, 'r')
    else:
        testfile = None
    outfile = prepfile(args.outfile, 'w')

    # Create feature extractor (you may want to modify this)
    fe = FeatureExtractor()

    # Read in training data
    train = DictReader(trainfile, delimiter='\t')

    # Split off dev section
    dev_train = []
    dev_test = []
    full_train = []

    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue
        feat = fe.features(ii['text'])
        if int(ii['id']) % 5 == 0:
            dev_test.append((feat, ii['cat']))
        else:
            dev_train.append((feat, ii['cat']))
        full_train.append((feat, ii['cat']))

    # Train a classifier
    sys.stderr.write("Training classifier ...\n")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)

    right = 0
    total = len(dev_test)
    for ii in dev_test:
        prediction = classifier.classify(ii[0])
        if prediction == ii[1]:
            right += 1
    sys.stderr.write("Accuracy on dev: %f\n" % (float(right) / float(total)))

    if testfile is None:
        sys.stderr.write("No test file passed; stopping.\n")
    else:
        # Retrain on all data
        classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)

        # Read in test section
        test = {}
        for ii in DictReader(testfile, delimiter='\t'):
            test[ii['id']] = classifier.classify(fe.features(ii['text']))

        # Write predictions
        o = DictWriter(outfile, ['id', 'pred'])
        o.writeheader()
        for ii in sorted(test):
            o.writerow({'id': ii, 'pred': test[ii]})
