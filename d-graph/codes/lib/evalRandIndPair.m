function ri = evalRandIndPair(bnrpred, pairSetTest, pairlabelTest)
%Inputs:
% bnrpred  - n x K binary prediction


    numpred= bnrlb2numlb(bnrpred);
 
 
    mlInd = pairlabelTest(:,1) == 1;
    ml1 = pairSetTest(mlInd,1);
    ml2 = pairSetTest(mlInd,2);
    
    clInd = pairlabelTest(:,1) == -1;
    cl1 = pairSetTest(clInd,1);
    cl2 = pairSetTest(clInd,2);
    
    mlA = numpred(ml1) == numpred(ml2);
    clA = numpred(cl1) ~= numpred(cl2);
    
    mlSum = sum(numpred(ml1) == numpred(ml2));
    clSum = sum(numpred(cl1) ~= numpred(cl2));
    
    disp(['ml = ' num2str(mlSum) ' out of ' num2str(length(ml1)) ]);
    disp(['cl = ' num2str(clSum) ' out of ' num2str(length(cl1)) ]);
    
    ri = 1. * (mlSum + clSum) / (length(ml1) + length(cl1));
 
end