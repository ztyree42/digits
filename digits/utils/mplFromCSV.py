import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


task = 'Classification'
path = '/home/ztyree/Downloads/run-Sep26_04-17-21_ip-172-31-33-229-tag-classifier_loss_'
test = pd.read_csv(path+'val.csv')
test['type'] = 'test'
train = pd.read_csv(path+'train.csv')
train['type'] = 'train'
df = pd.concat([train, test])
ax = sns.lineplot('Step', 'Value', hue='type', data=df)
ax.set(ylabel='Loss', title=f'Test Loss: {task}')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])
ax.get_figure().savefig('/home/ztyree/projects/digits/figs/classification.png', 
    dpi=2000, transparent=True)
