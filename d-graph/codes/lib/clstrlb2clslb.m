function clslb = clstrlb2clslb(clstrlb, label)

% clstrlb: N X K binary  cluster assignment matrix;
% label : N X 1 decimal  class labels;
% clslb:   N X 1 decimal class labels mapped from cluster assginment.

[N,K] = size(clstrlb);
clslb = zeros(N,1);

for iclstr = 1 : K
    insts = find(clstrlb(:, iclstr)==1);
    cls = mode(label(insts));
    clslb(insts) = cls;
end