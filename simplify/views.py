from django.shortcuts import render

# Create your views here.
from django.template.loader import get_template
from django.http import HttpResponse
from django.template import Context, loader
from django import forms
import os
import numpy as np
import time
from simplifier.settings import initializer
from meowls import *

class CommentForm(forms.Form):
    name = forms.CharField(initial='class')
    url = forms.URLField()
    comment = forms.CharField()


# LS_path = '/'

# command = 'python3 '+LS_path+'initializeLS.py'
# os.system(command)

def index(request):
    text_input=""
    context = {}


    lines = ""
    Sentence_Simplifier = request.GET.get('SSbtn')
    Sentence_Summarizer = request.GET.get('summarizebtn')
    Lexical_Simplifier = request.GET.get('LSbtn')

    if Sentence_Simplifier or Sentence_Summarizer or Lexical_Simplifier:

        text_input = request.GET.get('text_input')
        if text_input is not None:
            if Sentence_Simplifier:
                # execute this code
                print(text_input)

                f = open('model/input.txt', 'w')
                f.write(text_input)
                f.close()
                command = 'python model/nmt/translate.py -model model/sentence_simplifier.pt \
                                     -src model/input.txt\
                                     -output model/output.txt \
                                     -replace_unk \
                                     -beam_size 5'

                os.system(command)

            elif Sentence_Summarizer:

                text_input = request.GET.get('text_input')
                if text_input is not None:
                    f = open('model/input.txt', 'w')
                    f.write(text_input)
                    f.close()
                command = 'python3 model/Summarizer.py model/output.txt 0.4 model/input.txt'

                os.system(command)
            elif Lexical_Simplifier:
                subs_rank_nnclf = initializer.subs_rank_nnclf

                #########################################################
                substitutions_db = initializer.substitutions_db
                fivegram_model = initializer.fivegram_model
                threegram_model = initializer.threegram_model
                syllable_dict = initializer.syllable_dict
                wiki_frequency = initializer.wiki_frequency
                # word2vec = initializer.word2vec
                text_input = request.GET.get('text_input')
                print("LS input")
                if text_input is not None:
                    f = open('model/input.txt', 'w')
                    f.write(text_input)
                    f.close()
                # command = 'python3 meowls.py model/output.txt model/input.txt'
                # os.system(command)
                rank('model/input.txt', initializer)
        with open('model/output.txt', 'r') as f:
            for line in f.readlines():
                print(line)
                lines+=line

            f.close()
        context = {'text_input': text_input, 'text_output':lines}
    return render(request, 'template.html', context)

def summarize(request):
    text_input=""
    # if (request.GET.get('/summarize')):

    return render(request, 'summarize.html', {'text_input': text_input})
