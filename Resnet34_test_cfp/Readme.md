These script is used to calculate the verification accuracy on CFP dataset:

1.align the original CFP images to the size of 224x224 and adjust the list of frontal images and profile images.

2.Use the trained model to extract the 512 dim features.
(1)modify PYDIR,the trained model path and the gallery_list in run.sh according to your own settings;
(2)modify gpu_id in resnet34_cal_fea.py;
(3)set the gallery_list in run.sh to the CFP frontal images' list and run ./run.sh to extract and save the features of frontal images;
(3)set the gallery_list in run.sh to the CFP profile images' list and run ./run.sh to extract and save the features of profile images.

3.modify the CFP frontal list's path and profile list's path in get_fea_list.py and run get_fea_list.py to get the list of the saved features.

4.modify the CFP protocol's path and the saved features' path in create_mat_roc.py and run create_mat_roc.py to calculate the cosine similarity
between FF image pairs and FP image pairs.This will produce two mat file for each split(10 fold cross verification), one is cosine_similarity.mat 
including the cosine similarity for each image pair in the split, one is mated_label.mat indicating mated or not for each image pair in the split.

5.run the matlab script cal_acc-cfp.m to calculate the verification accuracy.
