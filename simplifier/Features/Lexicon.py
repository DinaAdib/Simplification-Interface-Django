from simplify.functions import *
"""**lexicon**"""

lexicons = {}
with open(DIRECTORY+"/lexicon.tsv") as f:
    for line in f:
        (key, val) = line.split()
        lexicons[key.lower()] = val