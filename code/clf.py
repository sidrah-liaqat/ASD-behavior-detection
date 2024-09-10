fold=3
subfold=0
operation = 'new'
debug=True# true for quick iterations during development
#sliding_inference = True # bool true or false, not string
learningRate = 0.0001
BATCHSIZE = 16
frame = 15
weightDecay = 0.0001
numEpoch = 100
thsEpoch = 99
save_in = 'lookface_tx_healthjour_fold{}'.format(fold)
BASEPATH = '/home/ubuntu/data/savedmodel/'
import os
os.makedirs(BASEPATH, mode=0o777, exist_ok=True)
PATH = BASEPATH + save_in + str(frame) + '.pth'
csv_PATH = save_in + str(frame) +'.csv'
active_learning = 0
# full15f = first overfitted model 15 frames
# full15f_1 = attempt at regularization (dropout, BN, lr increase)
action = 0
#TODO: overfitting
#TODO: augmentation
#TODO: aggregation on test set -  done uniform sliding window
#TODO: impact of dramatically large frame number
#TODO: Plot MSE ground truth vs clustering
#TODO: Train 15 frame, 20 frame and 40 frame transformer model
