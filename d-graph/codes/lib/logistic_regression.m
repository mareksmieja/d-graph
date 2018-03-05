function prob = logistic_regression( x, w )
% multiclass logistic resgression
% K : number of classes
% 
% Inputs: 
%  x - n x d data
%  w - K x (d+1) weight matrix 
% Outputs:
%  prob - n x K probability

K = size(w,1);
[n,~] = size(x);

x_aug = [x ones(n,1)];

A  = x_aug * w';
A_max = max(A, [], 2);
A = A - A_max(:,ones(K,1));
prob = exp(A);
Z_prob = sum(prob,2);
prob = prob./Z_prob(:,ones(K,1));
