from __future__ import print_function, division
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import FaceLandmarksDataset

from trans_form import ToTensor, RescaleLandmarks_test

from transformer_model import Transformer


from torch import optim, nn

# Ignore warnings
import warnings
import csv
warnings.filterwarnings("ignore")
from clf import *
# first training program
print("Training with {} frames on fold {}".format(frame, fold))
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#dev = torch.device("cpu")
loader_params_train = {'batch_size': BATCHSIZE, 'shuffle': True, 'num_workers': 0, 'drop_last':True}
loader_params_val = {'batch_size': BATCHSIZE, 'shuffle': False, 'num_workers': 0, 'drop_last':True}

# training data
train_dataset = FaceLandmarksDataset(csv_file_dir='/home/ubuntu/data/labels/',
                                    split='train',
                                    win=frame,
                                    transform = transforms.Compose([
                                        RescaleLandmarks_test(1),
                                        ToTensor()
                                    ]))

train_loader = DataLoader(dataset=train_dataset, **loader_params_train)

val_dataset = FaceLandmarksDataset(csv_file_dir='/home/ubuntu/data/labels/',
                                   split='val',
                                   win=frame,
                                   transform = transforms.Compose([
                                       RescaleLandmarks_test(1),
                                       ToTensor()
                                   ]))
val_loader = DataLoader(dataset=val_dataset, **loader_params_val)

class_weights = torch.FloatTensor([0.18, 0.82]).cuda()

criterion = nn.CrossEntropyLoss(weight=class_weights)


if operation == 'new':
    model = Transformer(k=76, heads=8, depth=8, seq_length=frame,
                                num_tokens=1000, num_classes=2)

    #print(model)

    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)  # 5e-6
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

elif operation == 'load':
    checkpoint = torch.load(BASEPATH + save_in + str(frame) + '_' + str(thsEpoch) + '.pth')

    epoch = checkpoint['epoch']
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']

model.to(dev)
loss_values = []
accuracy_values = []
val_acc_values = []
val_loss_values = []
min_val_loss = 1000

with open(BASEPATH+'training_'+save_in+str(frame)+'.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['epoch', 't_loss', 'wtd_t_acc', 'v_loss', 'wtd_v_acc',
                     'tr_sens', 'tr_spec', 'val_sens', 'val_spec', 'val_ppv'])
for epoch in range(numEpoch):
    print('Epoch {}'.format(epoch))
    training_loss = 0.0
    correct = 0
    tr_total = 0
    tr_true_pos = 0
    tr_total_pos = 0
    tr_true_neg = 0
    tr_total_neg = 0
    model.train()

    for tr_batch, tr_sample_batch in enumerate(train_loader):

        # get features and labels
        labels = tr_sample_batch['action']
        labels = labels.to(dev)

        f1 = tr_sample_batch['headpose'].float()
        f2 = tr_sample_batch['landmarks'].float()
        f3 = tr_sample_batch['eyelandmarks'].float()
        f4 = tr_sample_batch['au'].float()

        f1 = f1.to(dev)
        f2 = f2.to(dev)
        f3 = f3.to(dev)
        f4 = f4.to(dev)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(dev, v=f1, x=f2, y=f3, z=f4)
        # req size : v = (batch, frame, 6), x = (batch, frame, 137)
        # req size : y = (batch, size, 113), z = (batch, frame, 35)

        loss = criterion(output.permute(0,2,1), labels[:,:,action].long())

        loss.backward()
        optimizer.step()

        # print statistics
        training_loss += loss.item()
        correct += (torch.argmax(output, dim=2) == labels[:,:, action]).float().sum()

        tr_total += (labels.size(0)*labels.size(1))
        outputs_normalized = torch.softmax(output, dim=2)

        # since look face label is column zero, so the argmax operation below to get pred_labels is doing opposite of
        # expected. So need to do a 1-pred_labels operation
        predicted, pred_labels = torch.max(outputs_normalized, dim=2)

        tr_total_pos += (labels[:,:, action] == 1).sum()
        tr_total_neg += (labels[:,:, action] == 0).sum()

        
        tr_true_pos += torch.logical_and(((pred_labels) == labels[:,:, action]),
                                      (labels[:,:, action] == 1)).float().sum()
        tr_true_neg += torch.logical_and(((pred_labels) == labels[:,:, action]),
                                      (labels[:,:, action] == 0)).float().sum()

    print("Batch: " + str(tr_batch) + ", loss : " + str(training_loss*BATCHSIZE/len(train_dataset)))
    training_acc = 100 * correct / tr_total
    training_loss_scaled = 1000*training_loss*BATCHSIZE/len(train_dataset)

    tr_recall_pos = 100 * tr_true_pos / tr_total_pos
    tr_recall_neg = 100 * tr_true_neg / tr_total_neg

    tr_wtd_acc = ((100 * tr_true_pos / tr_total_pos) + (100 * tr_true_neg / tr_total_neg)) / 2

    plt.figure()

    ### validation loop
    model.eval()
    pred_label_arr = torch.Tensor(0).to(dev)
    label_arr = torch.Tensor(0).to(dev)
    with torch.no_grad():
        valcorrect = 0
        val_loss = 0
        total = 0
        true_pos = 0
        total_pos = 0
        false_pos = 0
        true_neg = 0
        total_neg = 0

        for val_batch, val_sample_batch in enumerate(val_loader):
            # get features and labels
            labels = val_sample_batch['action']
            labels = labels.to(dev)

            f1 = val_sample_batch['headpose'].float()
            f2 = val_sample_batch['landmarks'].float()
            f3 = val_sample_batch['eyelandmarks'].float()
            f4 = val_sample_batch['au'].float()

            f1 = f1.to(dev)
            f2 = f2.to(dev)
            f3 = f3.to(dev)
            f4 = f4.to(dev)

            outputs = model(dev, v = f1, x=f2, y=f3, z=f4)

            val_loss += criterion(outputs.permute(0, 2, 1), labels[:,:,action].long())
            outputs_normalized = torch.softmax(outputs, dim=2)
            pred_labels = torch.argmax(outputs_normalized, dim=2)
            predicted = outputs_normalized[:, :, 1]  # soft prediction scores

            pred_label_arr = torch.cat((pred_label_arr, pred_labels.float()))
            label_arr = torch.cat((label_arr, labels[:,:,action].float()))
            total += (labels.size(0)*labels.size(1))

            total_pos += (labels[:,:, action] == 1).sum()
            total_neg += (labels[:,:, action] == 0).sum()
            valcorrect += ((pred_labels == labels[:,:,action]).sum())

            true_pos += torch.logical_and(((pred_labels) == labels[:,:,action]),
                                          (labels[:,:, action] == 1)).float().sum()
            true_neg += torch.logical_and(((pred_labels) == labels[:,:,action]),
                                          (labels[:,:, action] == 0)).float().sum()
            false_pos += torch.logical_and((pred_labels) == 1, (labels[:,:, action] == 0)).float().sum()

    val_acc = 100 * valcorrect / total
    val_loss_scaled = 1000*val_loss*BATCHSIZE/len(val_dataset)

    recall_pos = 100 * true_pos / (total_pos + 1)
    recall_neg = 100 * true_neg / (total_neg + 1)
    precision = 100 * true_pos / (true_pos + false_pos + 1)
    val_wtd_acc = ((100 * true_pos / total_pos)+(100 * true_neg / total_neg))/2

    print('VAcc : %d %%' % val_acc)
    print('Vloss : {:.3f}'.format(val_loss_scaled/1000.0))
    print('Sensitivity : %d %%' % recall_pos)
    print('Specificity : %d %%' % recall_neg)
    print('Precision   : %d %%' % precision)
    print('Weighted average accuracy : %d %%' % val_wtd_acc)
    print('Total %d, P %d, N %d' % (total, total_pos, total_neg))


    with open(BASEPATH+'training_'+save_in+str(frame)+'.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        writer.writerow([str(epoch),#epoch
                        str(int(training_loss_scaled)),#training loss
                        str(int(tr_wtd_acc)),
                        str(int(val_loss_scaled)),# validation loss
                        str(int(val_wtd_acc)),# valication accuracy
                        str(int(tr_recall_pos)), # sensitivity
                        str(int(tr_recall_neg)), # specificity
                         str(int(recall_pos)),   # sensitivity
                         str(int(recall_neg)),   #precision
                         str(int(precision))])    
    # Training accuracy
    loss_values.append(training_loss / len(train_dataset))
    accuracy = correct / len(train_dataset)
    accuracy_values.append(accuracy)

    # Validation accuracy
    val_loss_values.append(val_loss / len(val_dataset))
    val_acc_values.append(valcorrect.item() / len(val_dataset))

    val_loss_now = val_loss / len(val_dataset)

    # Model checkpoint
    # removing the check for saving only minimum val loss model. Saving all models

    if val_loss_now < min_val_loss:
        min_val_loss = val_loss_now
        checkpoint = {
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler
            }
        PATH = BASEPATH + save_in + str(frame)  + '.pth'
        torch.save(checkpoint, PATH)
    """
    checkpoint = {
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler
    }
    PATH = BASEPATH + save_in + str(frame) + '_' + str(epoch) + '.pth'
    torch.save(checkpoint, PATH)
    """
    # update LR
    scheduler.step()

print("done")
