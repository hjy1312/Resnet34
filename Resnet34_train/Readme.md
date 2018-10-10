1.Preprocessing the data:
(1)align the original ms-1m images to 230x230, and put the data into the folder ./target_folder, 
the txt file training_list_without_deduplication.txt is the training list file in the order of
image_path1 ID_label1
image_path2 ID_label2
...
(2)Before training,crop the 230x230 images to 198x198 for removing part of background, and then
resize to 230x230,augment the data by random gray scaling,random horizontal flipping, color jittering
 and 224x224 random cropping. After these, transform the data to tensor and normalize its range to [-1,1].

2.Parameter setting:
(1)training epoch: By default it's 30;
(2)learning rate: the initial learning rate parameter is set to 0.1, and then decay to 0.05,0.01,0.001,0.0001 
respectively at epoch 5,10,15,20;
(3)weight_decay: 5e-4
(4)momentum: 0.9
(5)optimizer: SGD
(6)weight intialization: MSRA

3.Script description:
(1)dataset.py: define the dataset;
(2)model.py: define the Resnet34 model;
(3)resnet34_main.py: the main script to train resnet34;
(4)train.sh: the shell script to add the parser arguments and run the training;
(5)draw_loss_curve.py: read the log while training and plot the loss curve:
(6)net_ArcFace.py: define the arcface loss.

4.how to train:
(1)Environment Setting: Python 2.7 Anaconda version, Pytorch 0.3.0 or 0.3.1.
(2)modify train.sh according to your setting,including setting PYDIR to your python environment path and changing the parser arguments;
(3)modify the gpu_id setting in resnet34_main.py;
(4)modify the image path in the file training_list_without_deduplication.txt;
(5)run ./train.sh to start training.

