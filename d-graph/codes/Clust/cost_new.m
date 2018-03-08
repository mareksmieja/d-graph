function loss_value = cost_new(model, X_whole, indSet, constLabel, params, graphInd, graphVal)

    current_vector = [model.alphas model.bs];

    [N, D] = size(X_whole);
    K = params.max_class;
    
    
    weights = reshape(current_vector, K, D+1);
    
    if isequal(params.kernel,'linear')
        weights_vec = weights(:,1:D);
        energy = weights_vec(:)' * weights_vec(:);
    else
        % by convention labeled examples are ordered first
        alphas  = reshape(weights(1 : K*D), K, D);
    %     A_whole = alphas * [X_labeled;X_unlabeled]';
        A_whole = alphas * X_whole';
        energy = trace(A_whole*alphas');
    end
    
    p_y = logistic_regression(X_whole, weights);
    %X_aug = [X_whole ones(N, 1)];
    
    %koszt nielabelowany
    P = sum(p_y, 1)/N;    
    [balance_term,~] = sumsqr(P);
    
    
    %koszt nielabelowany
    ind1 = graphInd(:,1);
    ind2 = graphInd(:,2);
    probsGraph = p_y(ind1, :) .* p_y(ind2, :);
    sep_term = graphVal.' * sum(probsGraph, 2);

    %links
    [normalizer,~] = size(indSet);
    if normalizer == 0
        links = 0;
    else
        %koszt
        mlInd = constLabel(:,1) == 1;
        ml1 = indSet(mlInd,1);
        ml2 = indSet(mlInd,2);
        probsML = p_y(ml1, :) .* p_y(ml2, :);
        mlCost = sum(probsML(:));
        
        clInd = constLabel(:,1) == -1;
        cl1 = indSet(clInd,1);
        cl2 = indSet(clInd,2);
        probsCL = p_y(cl1, :) .* p_y(cl2, :);
        [n_cl,~] = size(cl1);
        clCost = n_cl - sum(probsCL(:));
        
        links = (mlCost + clCost) / (normalizer);
        
    end
    
    %ttt = 2.*(N - params.knn*params.max_class)/(params.max_class*N) - 1;
    ttt = 2./params.max_class - 1;
    loss_value = params.lambda * energy - ttt*balance_term - (1./(N*N))*sep_term - params.tau*links;
    
    
    
end
