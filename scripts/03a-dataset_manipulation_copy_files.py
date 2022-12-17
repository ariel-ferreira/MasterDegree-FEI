import os
import shutil

PATH_from = '/home/ferreiraa/Documents/Mestrado/agender_distribution/wav_traindevel'
PATH_to = '/home/ferreiraa/Documents/Mestrado/agender_distribution/feature_files'

source_folder = input()
#this code takes in the file path of our source folder\n",
destination_folder = input()
#this code takes in the file path of our destination folder"

a = 0
for folders, subfolders, filenames in os.walk(source_folder):
    for filename in filenames:
        if filename.endswith('{}'.format(extension)):
            shutil.copy(os.path.join(folders, filename), destination_folder)
            file = open("files_on_code.txt", 'a+')
            file.write(filename + '\\n')
            file.close()
 ]
},
{
 "cell_type": "code",
 "execution_count": 12,
 "metadata": {},
 "outputs": [],
 "source": [
  "d = os.path.join(folders, filename)"
 ]
},
{
 "cell_type": "code",
 "execution_count": 13,
 "metadata": {},
 "outputs": [
  {
   "data": {
    "text/plain": [
     "'/home/ferreiraa/Documents/Mestrado/agender_distribution/wav_traindevel/1417/2/a11417s18.raw'"
    ]
   },
   "execution_count": 13,
   "metadata": {},
   "output_type": "execute_result"
  }
 ],
 "source": [
  "d"
 ]
},
{
 "cell_type": "code",
 "execution_count": null,
 "metadata": {},
 "outputs": [],
 "source": [
  "import os\n",
  "import shutil\n",
  "\n",
  "src = r'C:\\TEMP\\dir'\n",
  "dest = r'C:\\TEMP\\new'\n",
  "\n",
  "for path, subdirs, files in os.walk(src):\n",
  "    for name in files:\n",
  "        filename = os.path.join(path, name)\n",
  "        shutil.copy2(filename, dest)"
 ]
},
{
 "cell_type": "code",
 "execution_count": null,
 "metadata": {},
 "outputs": [],
 "source": []
},
{
 "cell_type": "code",
 "execution_count": null,
 "metadata": {},
 "outputs": [],
 "source": []
}
,
metadata": {
"kernelspec": {
 "display_name": "Python 3.10.5 ('conv1d-venv')",
 "language": "python",
 "name": "python3"
},
"language_info": {
 "codemirror_mode": {
  "name": "ipython",
  "version": 3
 },
 "file_extension": ".py",
 "mimetype": "text/x-python",
 "name": "python",
 "nbconvert_exporter": "python",
 "pygments_lexer": "ipython3",
 "version": "3.10.5"
},
"vscode": {
 "interpreter": {
  "hash": "04cedfdd4aaeb3a14b7313c5139a1493a34cd254dccc2367acd55268f3eb1701"
 }
}
,
nbformat": 4,
nbformat_minor": 4
