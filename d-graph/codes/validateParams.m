function [values, acur_const_max] = validateParams_Pair(filename, per, X_whole, pairSet, pairlabel, params, alg, graphInd, graphVal)

tic

[~, dim] = size(X_whole);

lambdaset  = [0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125];
lamLen = length(lambdaset);
tauset  = [1.];
tauLen = length(tauset);

if per ==0 || isempty(pairSet)

    values.tau= 1.;
    values.lambda = 0.00390625/dim;
    acur_const_max = [];

else
    acur_const = zeros(lamLen, tauLen);
    for i = 1 : lamLen
        params.lambda = lambdaset(i)/dim;
        for j  = 1 : tauLen
            params.tau = tauset(j);
            acur_const(i,j) = crossValidateAccur(filename, X_whole, pairSet, pairlabel, params, alg, graphInd, graphVal);
        end
    end
    [acur_const_max, ind] = randIndMax(acur_const(:));

    [row,col] = ind2sub(size(acur_const),ind);
    values.lambda = lambdaset(row)/dim;
    values.tau = tauset(col);
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
    
    
        
    min_cost = realmax;
    for r = 1:params.repeat
        model = clust(X_whole, trainPair, trainLabel, params, graphInd, graphVal);
        current_cost = cost_new(model, X_whole, trainPair, trainLabel, params, graphInd, graphVal);
        if current_cost < min_cost
            min_cost = current_cost;
            optimal_model = model;
        end
    end

    weights = [optimal_model.alphas optimal_model.bs];
    prob = logistic_regression(X_whole, weights);
    [~, label] = max(prob, [], 2);
        
    predLabel = zeros(size(testLabel));
    predLabel(:,1) = label(testPair(:,1)) == label(testPair(:,2));
    
    acur(f) = sum(predLabel == testLabel)/testPairNum;

end
acur_mean = mean(acur);








