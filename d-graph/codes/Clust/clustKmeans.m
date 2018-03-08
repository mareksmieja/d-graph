function model = clustKmeans(X_whole, indSet, constLabel, params, graphInd, graphVal)


    [N, D] = size(X_whole);
    K = params.max_class;

    [params,options] = process_options(params);
    
    params.alpha_divisor = D;
    alphas = randn(K,D)/params.alpha_divisor;
    bs = randn(params.max_class,1)/params.alpha_divisor;%tu tez
    weights_old = [alphas bs];
    
    %kmeans
    weights_vec = weights_old(:);
    weights_vec = init_unsup(weights_vec, X_whole, options, params);
    weights_old = reshape(weights_vec,K,D+1);
    %kmeasn end
    
    result_vec = weights_old(:);

    result_vec = minFunc(@clust_cost_new,result_vec,options, X_whole, graphInd, graphVal, indSet, constLabel, params);
    
    weights_new = reshape(result_vec, size(weights_old));
    
    model.alphas =  weights_new(:, 1: end-1) ;
    model.bs = weights_new(:, end);

end


function result_vec = init_unsup(result_vec, X_unlabeled, options, params)

% options for optimization
opts = zeros(14, 1);
opts(5) = 1;
clist = X_unlabeled(1:params.max_class, :);

% K-means
[centres, error, post, errlog] = KmeansMetricWrapper(clist, X_unlabeled, opts);
d2 = dist2(X_unlabeled, centres);
[minvals, unsup_labels] = min(d2', [], 1);


% train supervised classifier on initial cluster labels
options.MAXITER = 50; % optimize loosely (few iterations)
params.display_terms = false;
params.tau = 0;
result_vec = minFunc(@rim_cost,result_vec,options,[],X_unlabeled',unsup_labels,params);
% result_vec = minFunc(@m_step_obj,result_vec,options,[],q_y, X_labeled, X_unlabeled, params);
end

