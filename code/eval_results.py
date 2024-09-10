import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clf import save_in
from clf import *
results = pd.read_csv(BASEPATH + save_in + str(frame) + 'test' +'results.csv')

age_range = np.array(['6', '9', '12', '18', '24', '36'])
#age_range = np.array(['18', '24', '36'])

all_split = pd.DataFrame(results.filename.str.split('_', 2).tolist(),
                                       columns=['sub_id', 'age', 'rest'])
mn = np.zeros(len(age_range))
std = np.zeros(len(age_range))
print("Age   Accuracy    Sensitivity    Specificity    Macro Avg Acc")
mn_acc = 0
mn_sens = 0
mn_spec = 0
mn_wtdacc = 0
for age in range(len(age_range)):
    #print(age_range[age])

    #df = results[results[all_split.age.isin([age]).index]]
    ind = list(np.where(all_split.age == age_range[age])[0])
    print("# examples : {}".format(len(ind)))
    df = results.iloc[ind]
    print("{}   {:.2f}  {:.2f}   {:.2f}   {:.2f}".format(age_range[age], df.accuracy.mean(),
                                                         df.sensitivity.mean(),
                                                         df.specificity.mean(),
                                                         df.macroavgacc.mean()
                                                         ))
    mn[age] = df.macroavgacc.mean()
    std[age] = df.macroavgacc.std()

    if(len(ind)>0):
        mn_acc += len(ind)*df.accuracy.mean()
        mn_sens += len(ind) * df.sensitivity.mean()
        mn_spec += len(ind) * df.specificity.mean()
        mn_wtdacc += len(ind) * df.macroavgacc.mean()
mn_acc = mn_acc/len(results)
mn_sens = mn_sens/len(results)
mn_spec = mn_spec/len(results)
mn_wtdacc = mn_wtdacc/len(results)

print("Overall Acc: {:.2f}   Sens: {:.2f}  Spec: {:.2f}   Macro Avg acc: {:.2f}".format
      (mn_acc, mn_sens, mn_spec, mn_wtdacc))

plt.figure(1)
plt.title('Weighted Accuracy')
plt.errorbar([ 6,9,12,18, 24, 36], mn, std,
                       linestyle='None', marker='^')
plt.show()

plt.figure(1)
plt.title('Weighted Accuracy')
plt.errorbar([ 6,9,12,18, 24, 36], mn, std,
                       linestyle='None', marker='^')
plt.show()

"""
plt.figure(2)
plt.title('Weighted Accuracy')
plt.plot(results.epoch,results.wtd_t_acc)
plt.plot(results.epoch, results.wtd_v_acc)
plt.show()

plt.figure(3)
plt.title('Recall of Positive class')
plt.plot(results.epoch,results.tr_sens)
plt.plot(results.epoch, results.tr_spec)
plt.show()

plt.figure(4)
plt.title('Recall of Negative class')
plt.plot(results.epoch,results.val_sens)
plt.plot(results.epoch, results.val_spec)
plt.show()
"""