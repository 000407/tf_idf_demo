from html.parser import HTMLParser
from html.entities import name2codepoint
from weight import Weight
import nltk
import re

nltk.download('punkt')

class MyParser(HTMLParser):
    weight = Weight.OTHER
    keydict = dict()
    ps = nltk.stem.PorterStemmer()

    def handle_starttag(self, tag, attrs):
        self.weight = Weight[tag.upper()] if tag.upper() in Weight.__members__ else Weight.OTHER

    def handle_data(self, data):
        self.stem(data)

    def get_keydict(self):
        return self.keydict

    def stem(self, data):
        corpus = nltk.tokenize.word_tokenize(data) #TODO: Might need to incorporate cleansing etc. for better results

        with open('cwords.txt', 'r') as cwords_txt:
            cwords = cwords_txt.read().split(',')
            
            keywords = [ item.lower() for item in corpus if item not in cwords ]
            
            for w in keywords:
                if re.match("^[a-z]{2,}$", w):
                    s = self.ps.stem(w)
                    if(s not in self.keydict):
                        self.keydict[s] = {'f': 0, 'w': {}}

                    if(self.weight.name not in self.keydict[s]):
                        self.keydict[s]['w'][self.weight.name] = 0

                    self.keydict[s]['f'] += 1
                    self.keydict[s]['w'][self.weight.name] += self.weight.value