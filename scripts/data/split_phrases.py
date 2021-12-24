"""
This script's purpose is to generate data for pyserini phrase indexing, and input file for a QA annotation tool.

It functions specificly to our needs:
- input is a .md file similar to 'data/uet_reg/qcdt.md'
- output: phrases.json, annotator-input.json
- splitting by \\n\\n
- add "header1. header2" to the begining of each phrases

Usage:$ python3 split_phrases.py <input_file> <output_dir>

"""

import json
import sys

input_file = sys.argv[1]
output_dir = sys.argv[2]

with open(input_file,'r',encoding='utf-8') as i:
  text = i.read()


# split by \n\n
phrases = text.split('\n\n')

# [optional] sliding windows for long phrases ()

# combines titles with phrase content
def is_title(text): return text[0] == '#'
def is_h1(text): return text[:2] == '# '
def is_h2(text): return text[:3] == '## '

final_phrases = []
squad = { "data": [] }

for i, phrase in enumerate(phrases):
  if not is_title(phrase):
    h1 = i
    while not is_h1(phrases[h1]): h1 -= 1
    h1 = phrases[h1][2:]

    h2 = i
    while not is_h2(phrases[h2]): h2 -= 1
    h2 = phrases[h2][3:]

    context = h1+".\n"+h2+".\n"+phrase

    final_phrases.append({
      "id": "qcdt_"+str(i),
      "contents": context,
    })

    squad["data"].append({
      "title": h2[:8]+" "+phrase[:40],
      "paragraphs": [{
        "context": context,
        "qas": []
      }]
    })


# dump
with open(output_dir+'phrases.json','w',encoding='utf-8') as o:
  json.dump(final_phrases, o, ensure_ascii=False, indent=2)

with open(output_dir+'annotator-input.json','w',encoding='utf-8') as o:
  json.dump(squad, o, ensure_ascii=False, indent=2)