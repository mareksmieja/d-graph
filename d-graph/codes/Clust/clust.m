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



function [params,options] = process_options(params)


% minFunc related
if ~isfield(params,'MAXITER')
    options.MAXITER = 400;
else
    options.MAXITER = params.MAXITER;
end
if ~isfield(params,'MAXFUNEVALS')
    options.MAXFUNEVALS = 2000;
else
    options.MAXFUNEVALS = params.MAXFUNEVALS;
end
if isfield(params,'Method')
    options.Method = params.Method;
else
    options.Method = 'lbfgs';
end
if isfield(params,'LS')
    options.LS = params.LS;
else
    options.LS = 4;
end
if isfield(params,'USEMEX')
    options.USEMEX = params.USEMEX;
else
    options.USEMEX = 0;
end

% options.Display = 'full'; %'excessive';
options.Display = 0; %'excessive';
end % process_options