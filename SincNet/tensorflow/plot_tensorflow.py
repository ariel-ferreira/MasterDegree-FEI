import re
import os
import pandas as pd
import matplotlib.pyplot as plt

def RemoveText(x):
    regex = '^\D*'
    x = re.sub(regex, '', x)
    return float(x)

def RemoveComma(x):
    regex = '\D'
    x = re.sub(regex, '', x)
    return int(x)

# Get the parent directory of the script
file_dir = os.path.dirname(__file__)

# Indicate the file that contain the results 
filename = 'results_rios_tensorflow_1st.txt'

# Read into a dataframe the txt file that contains the results 
results_df = pd.read_table(
             os.path.join(file_dir, filename), delimiter=' ', header=None)

del results_df[0]

results_df = results_df.rename(columns={1:'epochs'})
results_df = results_df.rename(columns={2:'acc_te'})
results_df = results_df.rename(columns={3:'acc_te_snt'})

results_df['epochs'] = results_df['epochs'].apply(lambda x: RemoveComma(x))
results_df['acc_te'] = results_df['acc_te'].apply(lambda x: RemoveText(x))
results_df['acc_te_snt'] = results_df['acc_te_snt'].apply(lambda x: RemoveText(x))

print(results_df)

plot = results_df.plot.line(x='epochs', y=['acc_te', 'acc_te_snt'])

plt.show()