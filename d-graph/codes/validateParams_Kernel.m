function [values, acur_const_max] = validateParams_Kernel(filename, per, X_whole, pairSet, pairlabel, params, alg)

tic

[~, dim] = size(X_whole);

lambdaset  = [0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125];
%lambdaset  = [0.00390625 0.001953125];
lamLen = length(lambdaset);

gammaset = [0.25 0.5 1.];
%gammaset = [0.5 1.];
gammaLen = length(gammaset);



if per ==0 || isempty(pairSet)

    values.tau= 1.;
    values.gamma = 1.;
    values.lambda = 0.00390625/dim;
    acur_const_max = [];

else
    acur_const = zeros(lamLen, gammaLen);
    for i = 1 : lamLen
        params.lambda = lambdaset(i)/dim;
        for j  = 1 : gammaLen
            params.gamma = gammaset(j);
            dataKernel = computeKernel(X_whole, params);
            [graphInd, graphVal] = findGraphTopKernelNN(dataKernel, params);
            acur_const(i,j) = crossValidateAccur(filename, dataKernel, pairSet, pairlabel, params, alg, graphInd, graphVal);
        end
    end
    [acur_const_max, ind] = randIndMax(acur_const(:));

    [row,col] = ind2sub(size(acur_const),ind);
    values.lambda = lambdaset(row)/dim;
    values.gamma = gammaset(col);
end

toc





function acur_mean = crossValidateAccur(filename, X_whole, pairSet, pairlabel, params, alg, graphInd, graphVal)

[N,D] = size(X_whole);
pairNum = size(pairSet,1);
F = 5;
if pairNum < F
    F = pairNum;
    for f = 1 : F
        fold{f} = f;
    end
else
    fold = crossvalidation(pairNum, F);
end

K = params.max_class;
acur = zeros(F,1);

for f  = 1 : F
    rng(params.seed);
    
    testInd = fold{f};
    
    testPair = pairSet(testInd,:);
    testPairNum = size(testPair,1);
    
    testLabel = pairlabel(testInd);
    testLabel(testLabel == -1) = 0;
    
    trainInd = setdiff(1:pairNum,testInd);
    trainPair = pairSet(trainInd,:);
    trainLabel = pairlabel(trainInd);
    
    
    model = clustKmeans(X_whole, trainPair, trainLabel, params, graphInd, graphVal);
    weights = [model.alphas model.bs];
    prob = logistic_regression(X_whole, weights);
    [~, label] = max(prob, [], 2); 
    
        
    predLabel = zeros(size(testLabel));
    predLabel(:,1) = label(testPair(:,1)) == label(testPair(:,2));
    
    acur(f) = sum(predLabel == testLabel)/testPairNum;

end
acur_mean = mean(acur);








