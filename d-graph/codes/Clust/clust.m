function model = clust(X_whole, indSet, constLabel, params, graphInd, graphVal)


    [N, D] = size(X_whole);
    K = params.max_class;

    [params,options] = process_options(params);
    
    params.alpha_divisor = D;
    alphas = randn(K,D)/params.alpha_divisor;
    bs = randn(params.max_class,1)/params.alpha_divisor;
    weights_old = [alphas bs];
    

    result_vec = weights_old(:);

    result_vec = minFunc(@clust_cost_new,result_vec,options, X_whole, graphInd, graphVal, indSet, constLabel, params);
    
    weights_new = reshape(result_vec, size(weights_old));
    
    model.alphas =  weights_new(:, 1: end-1) ;
    model.bs = weights_new(:, end);

end


