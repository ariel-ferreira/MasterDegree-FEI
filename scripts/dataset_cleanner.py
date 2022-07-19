import os
import glob

files = glob.glob('/home/ferreiraa/dataSet/audio/agender_distribution/wav_traindevel/**/*.wav.tmp', recursive=True)

for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))
