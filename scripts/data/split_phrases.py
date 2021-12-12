import json
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
#
# usage python3 norm.py <input_file> <output_file>
#

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

for i, phrase in enumerate(phrases):
  if not is_title(phrase):
    h1 = i
    while not is_h1(phrases[h1]): h1 -= 1
    h1 = phrases[h1][2:]

    h2 = i
    while not is_h2(phrases[h2]): h2 -= 1
    h2 = phrases[h2][3:]

    final_phrases.append({
      "id": "qcdt_"+str(i),
      "contents": h1+"\n"+h2+"\n"+phrase,
    })

# dump
with open(output_file,'w',encoding='utf-8') as o:
  json.dump(final_phrases, o, ensure_ascii=False, indent=2)