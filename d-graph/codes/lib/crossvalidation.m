function fold = crossvalidation(n, F)
% n     - number of instances
% F     - F fold

perFoldNum = floor(n/F);
rdInd = randperm(n);

fold = cell(F,1);

for i = 1 : F-1
    fold{i,1} = rdInd((i-1)*perFoldNum + 1 : i*perFoldNum); 
end

fold{F} = rdInd((F-1)*perFoldNum + 1 : end); 


