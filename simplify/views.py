from django.shortcuts import render

# Create your views here.
from simplifier.settings import *
from meowls import *
from complexity_ranker import *
from django.core.files.storage import FileSystemStorage


def get_complexity_scores(sent1 , sent2 , clf):
  _ , features1= get_complexity_features_sentence(sent1)
  score1= clf.predict(np.array(features1).reshape(1 , -1))
  _, features2 = get_complexity_features_sentence(sent2)
  score2= clf.predict(np.array(features2).reshape(1 , -1))
  return score1 , score2

def index(request):
    text_input=""
    lines = ""
    context = {}
    uploaded_file_url = ""


    Sentence_Simplifier = request.POST.get('SSbtn')
    Sentence_Summarizer = request.POST.get('summarizebtn')
    Lexical_Simplifier = request.POST.get('LSbtn')
    clear_all = request.POST.get('clearbtn')

    if request.POST.get('Upload'):
        if 'myfile' in request.FILES.keys():
            myfile = request.FILES['myfile']
            fs = FileSystemStorage()
            # filename = fs.save(myfile.name, myfile)
            # uploaded_file_url = fs.url(filename)
            for line in myfile:
                lines+= "".join( chr(x) for x in bytearray(line) )
            context = {'text_input': lines}#, 'uploaded_file_url':uploaded_file_url}
            return render(request, 'template.html', context)


    elif Sentence_Simplifier or Sentence_Summarizer or Lexical_Simplifier or clear_all:

        text_input = request.POST.get('text_input')

        if clear_all:
            text_input = ''
            lines = ''

        elif text_input is not None:
            text_input = text_input.rstrip('\n')
            if Sentence_Simplifier:
                ss_input = '\n'.join(nltk.sent_tokenize(text_input))
                f = open('model/input.txt', 'w')
                f.write(ss_input)
                f.close()
                command = 'python model/nmt/translate.py -model model/sentence_simplifier.pt \
                                     -src model/input.txt\
                                     -output model/output.txt \
                                     -replace_unk \
                                     -beam_size 5'

                os.system(command)

            elif Sentence_Summarizer:


                text_input = request.POST.get('text_input')
                if text_input is not None:
                    f = open('model/input.txt', 'w')
                    f.write(text_input)
                    f.close()
                command = 'python3 model/Summarizer.py model/output.txt 0.4 model/input.txt'

                os.system(command)

            elif Lexical_Simplifier:
                #########################################################

                text_input = request.POST.get('text_input')
                print("LS input")
                if text_input is not None:
                    f = open('model/input.txt', 'w')
                    f.write(text_input)
                    f.close()
                # command = 'python3 meowls.py model/output.txt model/input.txt'
                # os.system(command)
                rank('model/input.txt', '/model/output.txt', initializer)

            with open('model/output.txt', 'r') as f:
                for line in f.readlines():
                    print(line)
                    lines+=line

                f.close()
        # print(get_complexity_scores(text_input, lines, initializer.complexity_clf))

        context = {'text_input': text_input, 'text_output':lines}

    return render(request, 'template.html', context)

def summarize(request):
    text_input=""
    # if (request.GET.get('/summarize')):

    return render(request, 'summarize.html', {'text_input': text_input})