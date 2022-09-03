# Import packages
import os
from pydub import AudioSegment
from pydub.playback import play

DATASET_ROOT = os.path.join(os.path.expanduser("~"),
                            'dataSet/audio/agender_distribution/')
NETWORK_ROOT = os.path.join(os.path.expanduser("~"),
                            'Mestrado-PC/github/Conv1D/CNN/')

audio_file = "wav_traindevel/1001/2/a11001s10.wav"

# Play audio
playaudio = AudioSegment.from_file(os.path.join(DATASET_ROOT, audio_file), format="wav")
play(playaudio)