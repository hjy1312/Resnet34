function best_threshold=get_best_threshhold(root_dir,split)
max_acc = 0.0;
best_threshold = 0.0;
split_set = 1:10;
split_set(split) = [];

for threshold=-1:0.001:1
    sum_predict_right = 0.0;
    num = 0.0;
    for i=split_set
        split_dir = [root_dir,'/','verification_resnet34_split',num2str(i)];
        load([split_dir,'/','cosine_similarity.mat']);
        load([split_dir,'/','mated_label.mat']);
        predicted = score>threshold;
        predict_right = (predicted==label);
        sum_predict_right = sum_predict_right + sum(predict_right);
        num = num + length(label);
    end
    acc = sum_predict_right / num;
    if acc>max_acc
        max_acc = acc;
        best_threshold = threshold;
    end
end
end
