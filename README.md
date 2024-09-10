# Autism Spectrum Disorder related Behavior Detection from Videos of Infant Interaction
Model architecture for behavior detection from facial landmarks 

![smile detection.png](smile%20detection.png)

trainer.py : This python file contains code for model training. The training data is presented to the model through a construct called 'DataLoader'. The file 'data_loader.py' manages this. The dataloader also uses 'trans_form.py' for some preprocessing of the feature data. 

The transformer model is defined in 'transformer_model.py'. Some configuration settings are stored in 'clf.py'.

After the training is done, 'sand_box.py' is used for running inference on the trained look face model. Here 'test_data_loader.py' is used to read the feature data and labels from the csv files and presented to the model for inference on test set. Text will be output during the running of this script showing how the model is doing on each file in test split.

After you have run the model inference on the test set, you can run 'eval_results.py' for a more detailed analysis of how the model is doing per age group.

'analyze_training.py' : During or after the training, you can run this script to see the plots for loss curve, accuracy, and recall. Blue represents the metrics on training set and orange represents validation set.

--
Use 'python trainer.py' on command line to initiate model training, no additional arguments needed.

After completion of training, 'sand_box.py' can be used to run inference on the model. 

In clf.py, the parameter 'debug' can be set to True to train on a smaller dataset. To train model on complete data, set 'debug' to True.

--
Environment"
requirements.txt can be used to install conda environment.



