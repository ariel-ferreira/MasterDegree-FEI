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
filename = 'results_06112022.txt'

# Read into a dataframe the txt file that contains the results 
results_df = pd.read_table(
             os.path.join(file_dir, filename), delimiter=' ', header=None)

del results_df[0]

results_df = results_df.rename(columns={1:'epochs'})
results_df = results_df.rename(columns={2:'loss_tr'})
results_df = results_df.rename(columns={3:'err_tr'})
results_df = results_df.rename(columns={4:'loss_te'})
results_df = results_df.rename(columns={5:'err_te'})
results_df = results_df.rename(columns={6:'err_te_snt'})

results_df['epochs'] = results_df['epochs'].apply(lambda x: RemoveComma(x))
results_df['loss_tr'] = results_df['loss_tr'].apply(lambda x: RemoveText(x))
results_df['err_tr'] = results_df['err_tr'].apply(lambda x: RemoveText(x))
results_df['loss_te'] = results_df['loss_te'].apply(lambda x: RemoveText(x))
results_df['err_te'] = results_df['err_te'].apply(lambda x: RemoveText(x))
results_df['err_te_snt'] = results_df['err_te_snt'].apply(lambda x: RemoveText(x))

print(results_df)

plot = results_df.plot.line(x='epochs', y=['loss_tr', 'err_tr', 'loss_te', 'err_te', 'err_te_snt'])

plt.show()