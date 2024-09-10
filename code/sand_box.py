# Author : Sidrah Liaqat
# Was very helpful for developing the GUI for timeline and landmarks etc
# usage : python sand_box.py arg1
# Options for arg1 : test/asdtrain (which split to use)

from __future__ import print_function, division
import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision import transforms, utils
from test_data_loader import FaceLandmarksDataset
from util import show_landmarks, show_landmarks_pillow,\
    make_fullimage, mark_curr_frame, pred_to_stats
from trans_form import ToTensor, RescaleLandmarks, RescaleLandmarks_test
from active_learning import compute_score, enable_dropout
import imageio.v3 as imageio
from pathlib import Path
from scipy.special import xlogy
from sklearn.cluster import DBSCAN

# Ignore warnings
import warnings
from my_models import mlp_model
import csv
from clf import PATH, BASEPATH, frame, save_in, BATCHSIZE, action, thsEpoch, fold, subfold
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
print(PATH)
print(save_in)

sliding_inference = False
print('sliding inference {}'.format(sliding_inference))
if sliding_inference:
    append = 'with'
else:
    append = 'wout'
PREDICTION_PATH = '/home/ubuntu/data/predictions/'
os.makedirs(PREDICTION_PATH, mode=0o777, exist_ok=True)
VIDEO_PATH = 'video_out/' + 'videos'+ str(frame)+'/'
os.makedirs(VIDEO_PATH, mode=0o777, exist_ok=True)
AGGBASEPATH = '/home/sidrah/DL/synchrony/lookface_journal/'
LABEL_PATH = '/home/ubuntu/data/labels/'

agg_df_list = (pd.read_csv(file) for file in  [AGGBASEPATH +'aggregated_experimenter_synchrony_02132020.csv',
                                              AGGBASEPATH +'aggregated_parent_synchrony2_11102020.csv',
                                              AGGBASEPATH +'aggregated_parent_synchrony_02132020.csv'])
# Concatenate all DataFrames
aggregate = pd.concat(agg_df_list, ignore_index=True)

# making annotated video (eye gaze vector intersection with head) using imageio
# timeline labels also included in video

# overall sum for computing metrics on all videos - universal counters
o_valcorrect = 0
o_total = 0
o_total_pos = 0
o_total_neg = 0
o_true_pos = 0
o_true_neg = 0
o_false_pos = 0
o_pred_label_arr = torch.Tensor(0)
o_label_arr = torch.Tensor(0)
o_clust_label_arr = torch.Tensor(0)
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#dev=torch.device("cpu")
print('Loading model : {} epoch: {}'.format(PATH, thsEpoch))
#checkpoint = torch.load(PATH)
checkpoint = torch.load(BASEPATH+save_in+str(frame)+'.pth',  map_location=dev)
epoch = checkpoint['epoch']
model = checkpoint['model']
optimizer = checkpoint['optimizer']
scheduler = checkpoint['scheduler']
model.to(dev)
model.eval()

checksize = pd.read_csv('/home/sidrah/DL/synchrony/data_splits/fold{}/{}_fold{}_subfold{}.txt'.
                        format(fold,str(sys.argv[1]),fold,subfold), header=None)


checksize.dropna(inplace=True)
checksize.reset_index(drop=True, inplace=True)
split = str(sys.argv[1])+'/'

syndict = dict([('sy1e', 'Synchrony1 Examiner'),
                ('sy2e', 'Synchrony2 Examiner'),
                ('sy1p', 'Synchrony1 Parent'),
                ('sy2p', 'Synchrony2 Parent'),
                ('Synchrony', 'Synchrony')])
labeldict = dict([(0, 'look_face'),
                  (1, 'look_object'),
                  (2, 'smile'),
                  (3, 'vocal') ,
                  (4,'social_smile'),
                  (5,'social_vocal')])
inv_syndict = {v: k for k, v in syndict.items()}

data_rows_mse = []
data_rows_results = []

for xxx in range(checksize.shape[0]):
#for xxx in range(21, 30):

    my_file = Path(LABEL_PATH + checksize[0][xxx] + '.csv')
    if my_file.is_file() == False:
        print('File {} is not in the feature directory'.format(checksize[0][xxx] + '.csv'))
    else:
        face_dataset = FaceLandmarksDataset(csv_file_dir=LABEL_PATH,
                                            split=str(sys.argv[1]),
                                            win=frame,
                                            testind=xxx,
                                            sliding_inference=sliding_inference,
                                            transform = transforms.Compose([
                                                RescaleLandmarks_test(1),
                                                ToTensor()
                                            ]))
        loader_params_face = {'batch_size': BATCHSIZE, 'shuffle': False, 'num_workers': 0, 'drop_last':False}
        face_loader = DataLoader(dataset=face_dataset, **loader_params_face)

        input1_Size = frame*len(face_dataset.head_cols)
        input2_Size = (len(face_dataset.C_x_fc) + len(face_dataset.C_y_fc))
        input3_Size = (len(face_dataset.C_x_eye) + len(face_dataset.C_y_eye))
        input4_Size = (len(face_dataset.AU))

        #inputSize = len(face_dataset.C_x_fc) + len(face_dataset.C_y_fc) + len(face_dataset.head_cols)

        # one loop for getting inference on the model
        # as well as performance metrics
        start_idx = 0
        hist_arr = torch.Tensor(0).to(dev)
        with torch.no_grad():

            # per video counters
            valcorrect = 0
            val_loss = 0
            total = 0
            true_pos = 0
            total_pos = 0
            true_neg = 0
            false_pos = 0
            total_neg = 0
            predicted_arr = torch.Tensor(0).to(dev)
            pred_label_arr = torch.Tensor(0).to(dev)
            label_arr = torch.Tensor(0).to(dev)
            smile_arr = torch.Tensor(0).to(dev)
            socialsmile_arr = torch.Tensor(0).to(dev)
            lookface_arr = torch.Tensor(0).to(dev)
            num_pred = 6
            AL_predictions_arr = np.empty(shape=[0,2, num_pred])
            for i, sample in enumerate(face_loader):

                #print('Batch # {}, num of rows {} / {}'.format(i, i*BATCHSIZE*frame, face_dataset.landmarks_frame.shape[0]))
                labels = sample['action']
                labels = labels.to(dev)
                conf = sample['conf']
                conf = conf.to(dev)

                f1 = sample['headpose'].float()
                f2 = sample['landmarks'].float()
                f3 = sample['eyelandmarks'].float()
                f4 = sample['au'].float()

                f1 = f1.to(dev)
                f2 = f2.to(dev)
                f3 = f3.to(dev)
                f4 = f4.to(dev)

                #outputs = model(feat_head=f1, feat_lmk=f2, feat_eye=f3)
                outputs = model(dev, v=f1, x=f2, y=f3, z=f4)

                outputs_normalized = torch.softmax(outputs, dim=2)
                # pred_labels == 0 ---> look face, pred_labels == 1---> look object

                #predicted, pred_labels = torch.max(outputs_normalized, dim=2)
                # getting the max value of prediction for every row doesn't make any sense

                pred_labels = torch.argmax(torch.where(outputs_normalized > 0.5, 1, 0), dim=2)
                predicted = outputs_normalized[:,:,1]#soft prediction scores

                # when sliding_inference = True, the predictions can't just be stacked
                # Have to align them for each new time step and average them.
                #hist_arr = torch.cat((hist_arr, outputs_normalized[:,0]))
                predicted_arr = torch.cat((predicted_arr, torch.flatten(predicted).float()))
                pred_label_arr = torch.cat((pred_label_arr, torch.flatten(pred_labels).float()))
                label_arr = torch.cat((label_arr, torch.flatten(labels[:, :, action]).float()))
                lookface_arr = torch.cat((lookface_arr, torch.flatten(labels[:, :, 0]).float()))
                smile_arr = torch.cat((smile_arr, torch.flatten(labels[:, :, 2]).float()))
                socialsmile_arr = torch.cat((socialsmile_arr, torch.flatten(labels[:, :, 4]).float()))
                total += (labels.size(0) * labels.size(1))
                total_pos += (labels[:,:, action] == 1).sum()
                total_neg += (labels[:,:, action] == 0).sum()
                valcorrect += ((pred_labels == labels[:,:,action]).sum())

                true_pos += torch.logical_and(((pred_labels) == labels[:,:, action]),
                                              (labels[:,:, action] == 1)).float().sum()
                true_neg += torch.logical_and(((pred_labels) == labels[:,:, action]),
                                              (labels[:,:, action] == 0)).float().sum()
                false_pos += torch.logical_and((pred_labels) == 1, (labels[:,:, action] == 0)).float().sum()

                # check for video file ended
                # then print metrics for this file, save video, initialize new video

                # if condition checks whether video file has ended

                if (f1.shape[0] < BATCHSIZE) or (label_arr.shape[0] > face_dataset.filesize):
                    if i==0:
                        print('Size of feature file smaller than batch size')
                        continue
                    elif i>1:
                        print('{}-EOF'.format(label_arr.shape[0]))


                        # Precompute repeated expressions
                        true_pos_percentage = int(100 * true_pos / (total_pos + 1))
                        true_neg_percentage = int(100 * true_neg / (total_neg + 1))
                        average_percentage = (true_pos_percentage + true_neg_percentage) / 2

                        data_rows_results.append([
                                face_dataset.testfilename.replace('.mpg',''),
                                int(100 * valcorrect / total),
                                true_pos_percentage,
                                true_neg_percentage,
                                int(average_percentage)
                            ])

                    #label_arr = label_arr.repeat_interleave(frame)
                    #voc_arr = voc_arr.repeat_interleave(frame)
                    #pred_label_arr = pred_label_arr.repeat_interleave(frame)
                    earth_to_pred = pred_label_arr.cpu()
                    earth_to_label = label_arr.cpu()

                    # for saving predictions and ground truth labels
                    # Prepare the data in a list of rows
                    data_rows = []
                    for i in range(len(pred_label_arr)):
                        data_rows.append([i, earth_to_pred[i].item(), earth_to_label[i].item()])

                    # Write the list of rows to a CSV file
                    with open('/home/sidrah/DL/exp/data/prediction.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerows(data_rows)

                    ## Using DBSCAN for filling in missing predictions

                    pred_posidx = np.transpose(np.asarray(np.where(earth_to_pred==1)))
                    if pred_posidx.shape[0] == 0:
                        pred_posidx = np.ones((1, 1))

                    ## Using DBSCAN for filling in missing predictions
                    pred_clustered_arr = DBSCAN(eps=35, min_samples=8, metric='euclidean', metric_params=None,
                                                algorithm='auto', leaf_size=30, p=None, n_jobs=None).fit(pred_posidx)
                    #pred_clustered_arr = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None).fit(pred_label_arr.cpu().reshape(-1,1))
                    clust_posidx=pred_clustered_arr.labels_
                    n_clusters = len(set(clust_posidx)) - (1 if -1 in clust_posidx else 0)
                    clust = np.zeros_like(earth_to_pred)

                    # process the cluster sets such that for each new set, make all the elements within that index range 1

                    pred_posidx = np.squeeze(pred_posidx, axis=1)
                    for i in range(n_clusters):
                        this_cluster = np.where(clust_posidx==i)
                        #print('min'+str(pred_posidx[np.min(this_cluster)]))
                        #print('max'+str(pred_posidx[np.max(this_cluster)]))
                        clust[pred_posidx[np.min(this_cluster)]:pred_posidx[np.max(this_cluster)]]=1

                    n_clusters_ = len(set(clust_posidx)) - (1 if -1 in clust_posidx else 0)

                    static_img = make_fullimage(lookface_arr.to(torch.device("cpu")),

                                                smile_arr.to(torch.device("cpu")),
                                                clust,
                                                socialsmile_arr.to(torch.device("cpu")))
                    static_img = Image.fromarray(static_img, 'RGB')
                    draw = ImageDraw.Draw(static_img, 'RGBA')
                    image = np.asarray(imageio.imread('/home/ubuntu/data/videos/' + face_dataset.testfilename,
                                                      index=face_dataset.landmarks_frame.loc[100, 'frame']))

                    label_arr_np = np.asarray(label_arr.to(torch.device("cpu")))
                    # writing metrics on image
                    draw.text((50, 660), 'GndTr: Dur: {:0.2f}, Freq: {}'.format(pred_to_stats(label_arr_np)[0],
                                                                                pred_to_stats(label_arr_np)[1]))
                    draw.text((50, 680), 'Clust: Dur: {:0.2f}, Freq: {}'.format(pred_to_stats(clust)[0],
                                                                                pred_to_stats(clust)[1]))
                    draw.text((600, 640), 'Look Face GT')
                    draw.text((600, 680), 'Smile GT')
                    draw.text((600, 660), 'Look Face Pred')
                    draw.text((600, 700), 'Social smile GT')
                    draw.text((50, 700), 'Prec: {:.2f}, Sens: {:.2f}, Spec: {:.2f}, Macro ave Acc: {:.2f}'.format(
                        (100 * true_pos / (true_pos + false_pos)),
                        (100 * true_pos / (total_pos + 1)), (100 * true_neg / total_neg),
                        (((100 * true_pos / (total_pos + 1)) + (100 * true_neg / total_neg)) / 2)),
                              (255, 255, 255), )
                    os.makedirs('video_out/look_face/frame15', mode=0o777, exist_ok=True)
                    imageio.imwrite('video_out/look_face/frame15/' + face_dataset.testfilename.replace('.mpg','') + '.png',
                                    np.vstack((image, static_img)))

                    # print freq and duration of label_arr and clust
                    print('GndTr: Dur: {:0.2f}, Freq: {}'.format(pred_to_stats(label_arr_np)[0], pred_to_stats(label_arr_np)[1]))
                    print('Clust: Dur: {:0.2f}, Freq: {}'.format(pred_to_stats(clust)[0], pred_to_stats(clust)[1]))

                    data_rows_mse.append([face_dataset.testfilename.replace('.mpg',''),
                                        #round(dur_agg, 2),
                                        #round(fq_agg, 2),
                                        round(pred_to_stats(label_arr_np)[0], 2),
                                        round(pred_to_stats(label_arr_np)[1], 2),
                                        round(pred_to_stats(clust)[0], 2),
                                        round(pred_to_stats(clust)[1], 2)
                                        ])

                    # for saving clustered predictions for pipeline
                    os.makedirs(PREDICTION_PATH + 'look_face_model/' + split, mode=0o777, exist_ok=True)
                    with open(PREDICTION_PATH + 'look_face_model/' + split + face_dataset.testfilename.replace('.mpg','.csv'), 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(['time','look_face'])
                        for i in range(len(clust)):
                            writer.writerow([str(i/30.0),str(clust[i].item())])


                    # uncomment for saving video with features and labels
                    """
                    #w = imageio.get_writer(VIDEO_PATH + face_dataset.testfilename, duration=33)
                    w = []
    
                    #reader = imageio.get_reader(
                    #    '/home/ubuntu/data/videos/' + face_dataset.testfilename,
                    #    'ffmpeg')
                    for idx in range(start_idx,face_dataset.landmarks_frame.shape[0]):
                        abs_idx = idx - start_idx
                        image = np.asarray(imageio.imread('/home/ubuntu/data/videos/' + face_dataset.testfilename,
                                                          index=face_dataset.landmarks_frame.loc[abs_idx if abs_idx < i*frame - start_idx else start_idx, 'frame'],
                                                          plugin = "pyav"))
                        # the face_dataset[] should be used every frame-th iteration to fetch #frame more features, idx starting index
                        if idx % frame == 0:
                            full_sample = face_dataset[int(idx / frame)]
                            lmk_temp = full_sample['landmarks'].unsqueeze(0).view(frame, -1, 2)
                            eye_temp = full_sample['eyelandmarks'].unsqueeze(0).view(frame, -1, 2)
                            head_temp = full_sample['headpose'].unsqueeze(0).view(frame, -1, 2)
                            #gaze_temp = full_sample['gazeangle'].unsqueeze(0).view(10, -1, 2)
                            ctr = 0
    
                        this_sample = {
                            'landmarks': lmk_temp[ctr],
                            'eyelandmarks': eye_temp[ctr],
                            'headpose': head_temp[ctr],
                            #'gazeangle': gaze_temp[ctr]
                        }
                        ctr = ctr + 1
    
                        ann_img = show_landmarks_pillow(this_sample, image)
    
                        timeline_img = make_fullimage(voc_arr.to(torch.device("cpu")),
                                                      label_arr.to(torch.device("cpu")),
                                                      pred_label_arr.to(torch.device("cpu")),
                                                      clust)
                        timeline_out = mark_curr_frame(timeline_img, idx-start_idx)
    
                        w.append(np.array(np.vstack((ann_img, timeline_out))))
    
                    imageio.imwrite(VIDEO_PATH + face_dataset.testfilename.replace('.mpg','.mp4'), w, quality=8, fps=30)
                    """
                    # updating the universal counters before zeroing the per video counters
                    o_valcorrect += valcorrect
                    o_total += total
                    o_total_pos += total_pos
                    o_total_neg += total_neg
                    o_true_pos += true_pos
                    o_true_neg += true_neg
                    o_false_pos += false_pos
                    o_pred_label_arr = torch.cat((o_pred_label_arr, pred_label_arr.float().cpu()))
                    o_label_arr = torch.cat((o_label_arr, label_arr.float().cpu()))
                    o_clust_label_arr = torch.cat((o_clust_label_arr, torch.from_numpy(clust)))

                    ###
                    valcorrect = 0
                    val_loss = 0
                    total = 0
                    true_pos = 0
                    total_pos = 0
                    true_neg = 0
                    total_neg = 0
                    #pred_label_arr = torch.Tensor(0).to(dev)
                    #label_arr = torch.Tensor(0).to(dev)
                    break

        with open(BASEPATH + save_in + str(frame) + str(sys.argv[1]) + 'results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['filename', 'accuracy', 'sensitivity', 'specificity', 'macroavgacc'])
            writer.writerows(data_rows_results)

        with open(BASEPATH + save_in + str(frame) + str(sys.argv[1]) + 'mse.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['filename', 'agg_dur', 'agg_fq', 'gt_dur', 'gt_fq', 'clust_dur', 'clust_fq'])
            writer.writerows(data_rows_mse)

        #print("pausehere")
print("Overall----")
print('Accuracy : %d %%' % (100 * o_valcorrect / o_total))
print('Sensitivity : %d %%' % (100 * o_true_pos / (o_total_pos + 1)))
print('Specificity : %d %%' % (100 * o_true_neg / (o_total_neg + 1)))
print('Precision : %d %%' % (100 * o_true_pos / (o_true_pos + o_false_pos)))
print('Weighted average accuracy : %d %%' % (
        ((100 * o_true_pos / (o_total_pos + 1)) + (100 * o_true_neg / (o_total_neg + 1))) / 2))
print('Total %d, P %d, N %d' % (o_total, o_total_pos, o_total_neg))

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
print(classification_report(o_label_arr, o_pred_label_arr))
auc = roc_auc_score(o_label_arr, o_pred_label_arr)
print('ROC AUC: %f' % auc)
print('After clustering')
print(classification_report(o_label_arr, o_clust_label_arr))
auc = roc_auc_score(o_label_arr, o_clust_label_arr)
print('ROC AUC (clustered): %f' % auc)
print("pauseagain")