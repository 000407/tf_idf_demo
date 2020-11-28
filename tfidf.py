from functools import reduce
import numpy as np
import json

class TfIdf():
	def __init__(self, docs):
		self.docs = docs

	def tf(self, term, corpus_id):
		frequencies = [ d['f'] for d in self.docs[corpus_id]['keywords'].values() ]

		return self.docs[corpus_id]['keywords'][term]['f'] / reduce(lambda a, b: a + b, frequencies)
		
	def idf(self, term):
		docs_with_term = [ d for d in self.docs.values() if term in d['keywords'] ]

		df = len(docs_with_term)
		# Not sure of which of the following two is correct
		# return np.log(len(self.docs) / (df + 1)) 
		return np.log( (len(self.docs) + 1) / (df + 1) ) + 1

	def tf_idf(self, term, corpus_id):
		weights = self.docs[corpus_id]['keywords'][term]['w'].values()
		w = reduce(lambda a, b: a + b, weights)

		return self.tf(term, corpus_id) * self.idf(term)
