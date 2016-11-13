import numpy as np
import pandas as pd
import scipy.stats as sc

df_Keras = pd.read_csv('submission_keras_shift_perm.csv')
df_Xgboost = pd.read_csv('sub_v_5_long2_python.csv')

df = pd.DataFrame({'id': df_Keras['id'], 'loss': df_Keras['loss']*0.4+df_Xgboost['loss']*0.6})
df.to_csv('Ensemble All State.csv', index = False)

#n, bins, patches = plt.hist(df_Keras['loss'], 50, normed=1, facecolor='green', alpha=0.75)
