 function [ acur_ari, acur_cls, acur_f1, acur_ri] = evalAccur(bnrpred, y)
%Inputs:
% bnrpred  - n x K binary prediction
% y        - n x 1 numerical true label
%Outputs:
% accur_rd - rand index accuracy
% accur_cls - purity accuracy
% accur_nmi - normalized mutual information

% Remove redundant label
 labelset = unique(y);
 lablen = length(labelset);
 label = zeros(size(y));
 for ilab = 1 : lablen
     label( y == labelset(ilab) ) = ilab;
 end

 numpred= bnrlb2numlb(bnrpred);
 acur_ari  = accuracy_ari(numpred, label); %Adjusted Rand Index
%  
%  bnrlb = numlb2bnrlb(y);
%  acur_ri = accuracy_ri(bnrpred , bnrlb);
%  
 numpredcls  = clstrlb2clslb(bnrpred, label);
 acur_cls =  accuracy_purity(numpredcls, label); % Purity

%  acur_nmi =  nmi_1(label, numpredcls); % Normalized Mutual Informaition
 
 [acur_ri, acur_f1] = reportAccuracy( label, numpred ); 