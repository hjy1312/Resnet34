clear; 
%clc;
root_dir = './fea_cfp/FF';
acc = 0.0;
for split=1:10
save_dir = './result_cfp_FF';
if ~exist(save_dir,'dir')
    mkdir(save_dir)
end
split_dir = [root_dir,'/','verification_resnet34_split',num2str(split)];
load([split_dir,'/','cosine_similarity.mat']);
load([split_dir,'/','mated_label.mat']);
threshold = get_best_threshhold(root_dir,split);
acc = acc + calculate_accuracy(score,label,threshold);
end
acc = acc / 10.0;
fprintf([root_dir,': ',num2str(acc*100),'%%']);fprintf('\n');

root_dir = './fea_cfp/FP';
acc = 0.0;
for split=1:10
%split = 9;
save_dir = './result_cfp_FP';
if ~exist(save_dir,'dir')
    mkdir(save_dir)
end
split_dir = [root_dir,'/','verification_resnet34_split',num2str(split)];
load([split_dir,'/','cosine_similarity.mat']);
load([split_dir,'/','mated_label.mat']);
threshold = get_best_threshhold(root_dir,split);
acc = acc + calculate_accuracy(score,label,threshold);
end
acc = acc / 10.0;
fprintf([root_dir,': ',num2str(acc*100),'%%']);fprintf('\n');
