function acc=calculate_accuracy(score,label,threshold)
predicted = score>threshold;
predict_right = (predicted==label);
acc = sum(predict_right) / length(label);
end
