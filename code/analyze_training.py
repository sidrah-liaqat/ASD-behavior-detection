import pandas as pd
import matplotlib.pyplot as plt
from clf import *
results = pd.read_csv(BASEPATH+'training_'+csv_PATH)

import matplotlib

matplotlib.use('module://backend_interagg')

plt.figure(1)
plt.title('Loss')
plt.plot(results.epoch,results.t_loss)
plt.plot(results.epoch, results.v_loss)
plt.show()

plt.figure(2)
plt.title('Weighted Accuracy')
plt.plot(results.epoch,results.wtd_t_acc)
plt.plot(results.epoch, results.wtd_v_acc)
plt.show()

plt.figure(3)
plt.title('Recall of Positive class')
plt.plot(results.epoch,results.tr_sens)
plt.plot(results.epoch, results.val_sens)
plt.show()

plt.figure(4)
plt.title('Recall of Negative class')
plt.plot(results.epoch,results.tr_spec)
plt.plot(results.epoch, results.val_spec)
plt.show()