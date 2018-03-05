function [loss_value, the_grad] = clust_cost_new(weights_vec, X_whole, graphInd, graphVal, indSet, constLabel, params)

    [N, D] = size(X_whole);
    K = params.max_class;
    
    weights = reshape(weights_vec, K, D+1);
    
    if isequal(params.kernel,'linear')
        energy = weights_vec(1 : K*D)' * weights_vec(1 : K*D);
        grad_energy = 2 * weights;
        grad_energy(:, end) = 0; % last column is the gradient of bs;
    else
        alphas  = reshape(weights_vec(1 : K*D), K, D);
        A_whole = alphas * X_whole';
        energy = trace(A_whole*alphas');
        grad_energy = 2*A_whole;
        grad_energy = [grad_energy zeros(size(grad_energy,1), 1)];
    end
    
    p_y = logistic_regression(X_whole, weights);
    X_aug = [X_whole ones(N, 1)];
    
    %kunlabeled cost 1
    P = sum(p_y, 1)/N;    
    [balance_term,~] = sumsqr(P);
    
    
    
    %unlabeled cost 2
    ind1 = graphInd(:,1);
    ind2 = graphInd(:,2);
    probsGraph = p_y(ind1, :) .* p_y(ind2, :);
    sep_term = graphVal.' * sum(probsGraph, 2);
    
    
    
    %gradient unlabeled 1
    rep_P = repmat(P, N, 1);
    S_mat = sum(p_y .* rep_P,2);
    rep_S = repmat(S_mat, 1, params.max_class);
    tmp_mat = p_y .* (rep_P - rep_S);
    grad_balance = 2 * tmp_mat.' * X_aug / N;
    

    %gradient unlabeled 2
    weightProbs = repmat(graphVal, 1, K) .* probsGraph;
    weightProbsRep = repmat(sum(weightProbs, 2), 1, K);
    grad_sep = weightProbs.' * (X_aug(ind1, :) + X_aug(ind2, :)) - (weightProbsRep .* p_y(ind1,:)).' * X_aug(ind1, :) - (weightProbsRep .* p_y(ind2,:)).' * X_aug(ind2, :);
    
    
    %links
    [normalizer,~] = size(indSet);
    if normalizer == 0
        links = 0;
        grad_links = 0;
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
        
        %gradient
        r_P_ML = repmat(sum(probsML, 2), 1, K);
        g_ML = probsML.' * (X_aug(ml1, :) + X_aug(ml2, :)) - (r_P_ML .* p_y(ml1,:)).' * X_aug(ml1, :) - (r_P_ML .* p_y(ml2,:)).' * X_aug(ml2, :);
        r_P_CL = repmat(sum(probsCL, 2), 1, K);
        g_CL = probsCL.' * (X_aug(cl1, :) + X_aug(cl2, :)) - (r_P_CL .* p_y(cl1,:)).' * X_aug(cl1, :) - (r_P_CL .* p_y(cl2,:)).' * X_aug(cl2, :);
        
        grad_links =  (g_ML - g_CL) / (normalizer);
    end
    
    %ttt = 2.*(N - params.knn*params.max_class)/(params.max_class*N) - 1;
    ttt = 2./params.max_class - 1;
    loss_value =  params.lambda * energy - ttt*balance_term - (1./(N*N))*sep_term - params.tau*links;
    the_grad = params.lambda * grad_energy - ttt*grad_balance - (1./(N*N))*grad_sep  - params.tau*grad_links;
    the_grad = the_grad(:);
    
end

