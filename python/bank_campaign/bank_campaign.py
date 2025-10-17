import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

raw_data = pd.read_csv('Example-bank-data.xls')
raw_data['y'] = raw_data['y'].map({'yes':1, 'no':0})
yes = raw_data[raw_data['y']>0]
no = raw_data[raw_data['y']<1]
f,axs = plt.subplots(1,2, sharex='row',figsize=(16,8))
axs[0].set_title('No Subscription')
axs[0].hist(no)
axs[1].set_title('Subscription')
axs[1].hist(yes)
plt.show()
