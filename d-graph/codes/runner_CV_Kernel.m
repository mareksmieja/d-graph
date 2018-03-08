function runner_CV_Kernel(alg, datas)
    
    %this code uses minFunc library from https://www.cs.ubc.ca/~schmidtm/Software
    addpath(genpath('./'), genpath('../../minFunc_2012/') );
    params.repeat = 50;
    params.seed = 0;
        
    
    alg = 'd-graph';
    params.knn = 5;
    params.kernel = 'RBF';
    
    
    filenames = {'vertebral'};
    scaled = '-scaled';
    scenarios = {'0', '0.05', '0.1', '0.15', '0.2'};
    samples = [0,1];
    
    loaddir = '../data/';
    resdir = 'res/';
    for filename = filenames
        scenariosLen = length(scenarios);
        samplesLen = length(samples);
        resultsy = zeros(scenariosLen, samplesLen);
        resyFile = char(strcat(loaddir, resdir, alg, '-', num2str(params.knn), '-', filename, scaled, '-ARI.out'));
        
        if exist(resyFile, 'file')==2
            delete(resyFile);
        end
        
        %% load data and constraints

        ldfile = char(strcat(loaddir, filename, scaled, '.in'));
        data = load(ldfile);
        X_whole = data(:,1:end-1);
        Y_whole = data(:, end);
        
        
        for samInd = 1 : samplesLen
            sam = samples(samInd);
            
            ariFile = char(strcat(loaddir, resdir, alg, '-', num2str(params.knn), '-',filename, scaled, '-', int2str(sam), '-ARI.out'));
            if exist(ariFile, 'file')==2
                delete(ariFile);
            end
            for scenInd = 1 : scenariosLen
                scenario = scenarios(scenInd);

                %initialize random generator - the same for all scenarios
                rng(params.seed + sam);
                


                if strcmp(scenario, '0')
                    pairSet = [];
                    pairlabel = [];
                else
                    pairfile = char(strcat(loaddir, filename, '_', scenario, '_links_', int2str(sam), '.in'));
                    disp(pairfile);
                    pairData = load(pairfile);
                    pairSet = pairData(:,1:end-1) + 1;
                    pairlabel = pairData(:,end);
                end
                params.max_class = length(unique(Y_whole));

                tic

                % --------------------------------- methods -----------------------------------------------

                disp('Cross Validation....');

                params.tau  = 1.; %fixed tau

                [values, ~] =  validateParams_Kernel(filename, str2double(scenario), X_whole, pairSet, pairlabel, params, alg);
                params.lambda = values.lambda;
                params.gamma = values.gamma;


                params.paramsNo1 = params.lambda;
                params.paramsNo2 = params.gamma;

                
                
                dataKernel = computeKernel(X_whole, params);
                [graphInd, graphVal] = findGraphTopKernelNN(dataKernel, params);
                
                
                optimal_model = clustKmeans(dataKernel, pairSet, pairlabel, params, graphInd, graphVal);
                
                weights = [optimal_model.alphas optimal_model.bs];
                prob = logistic_regression(dataKernel, weights);
                [~, numpred] = max(prob, [], 2);

                toc

                % ***** evaluate *********
                bnrpred = numlb2bnrlb(numpred);

                [ acur_ari, ~, ~, ~ ] = evalAccur(bnrpred, Y_whole);
                

               
                C = [str2double(scenario) acur_ari params.paramsNo1 params.paramsNo2];
                dlmwrite(ariFile, C, '-append');
                disp(['Labeled = ' scenario 'Acc = ' num2str(acur_ari)]);
                resultsy(scenInd, samInd) = acur_ari;
            end
        end
        means = mean(resultsy, 2);
        stds = std(resultsy, 0, 2);
        for scenInd = 1 : scenariosLen
            C = [str2double(scenarios(scenInd)) means(scenInd) stds(scenInd)];
            dlmwrite(resyFile, C, '-append');
        end
    end
end






