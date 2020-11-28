#!/usr/bin/python3

from mparser import MyParser
from tfidf import TfIdf
from requests import get
from collections import OrderedDict
import json
import hashlib
from weight import Weight

docs = dict() # Web Directory

with open('URLList.txt') as file:
	for url in file:
		parser = MyParser()

		print("Getting data from " + url.strip() + "...", end="", flush=True)
		response = get(url=url)

		print("done!\nParsing HTML data...", end="", flush=True)
		parser.feed(response.text)
		print("done!")

		keydict = parser.get_keydict()

		urldata = {
			"url": url,
			"keywords": keydict
		}

		id_md5 = hashlib.md5(url.encode()).hexdigest()

		docs[id_md5] = urldata

ti = TfIdf(docs)

for kd, d in docs.items():
	print("Processing document " + kd + "...", end="", flush=True)
	for kw, t in d['keywords'].items():
		docs[kd]['keywords'][kw]['tf_idf'] = ti.tf_idf(kw, kd)
	print("done!")

fname = 'webdirectory.txt'
print("Saving to file " + fname + "...", end="", flush=True)
with open(fname, 'w') as file:
	file.write(json.dumps(docs, sort_keys=False))

print("done!\nCompleted!")
