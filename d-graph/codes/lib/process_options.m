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