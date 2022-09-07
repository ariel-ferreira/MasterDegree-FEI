import os
import glob
import re

# Habilitar sessão para renomear arquivos

files = glob.glob('/home/ferreiraa/dataSet/audio/agender_distribution/wav_traindevel/**/*.wav-n.wav', recursive=True)

for f in files:
    try:
        fo = re.sub('.wav-n.wav', '-n.wav', f)
        command = 'cp '+ f + ' ' + fo
        os.system(command)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))

# Habilitar sessão para remover arquivos

files = glob.glob('/home/ferreiraa/dataSet/audio/agender_distribution/wav_traindevel/**/*.wav-n.wav', recursive=True)

for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))