function accur = accuracy_purity(predict, label)

 accur = sum(predict==label)./length(label);   