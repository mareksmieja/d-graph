function acur = accuracy_ri( clustLb, realLb)
% 
% Clustering Rand Index Accuracy
% clustLb  - binary matrix indicating clustering result
% realLb   - binary matrix indication underlying classes
% K  - number of clusters
%
%   The accuracy is given by the following measurement
%   
%   accuracy = sum_i>j (1{1{c_i == c_j} == 1{realc_i == realc_j} )/
%                               0.5*(K-1)*K


 [N, ~] = size(clustLb);

% pairs = 0.5 * (N-1) *N;
% 
% os = find( clustLb == 0);
% clustLb(os) = -1;
% 
% os = find( realLb == 0);
% realLb(os) = -1;
% 
% agreeNum =0;
% 
% for i = 1: N-1
%     iclstlb = repmat( clustLb(i,:), N-i, 1);
%     neibs = iclstlb .* clustLb(i+1:N,:);
%     neibs = sum(neibs,2);
%         
%     ireallb = repmat( realLb(i,:), N-i, 1);
%     realneibs = ireallb .* realLb(i+1:N,:);
%     realneibs = sum(realneibs,2);
%    
%     agreeNum = agreeNum + length(find(neibs == realneibs));
%     
% end
% 
% acur = agreeNum/pairs;

Q_pred  = clustLb * clustLb';
Q_label = realLb  * realLb';

acur = nnz(Q_pred==Q_label)/(N^2);
