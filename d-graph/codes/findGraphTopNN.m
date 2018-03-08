function [graphInd, graphVal] = findGraphTopNN(X_whole, params)
    [N,D] = size(X_whole);
    params.gammaHeur = heuristic_gamma(D);
    kk= params.knn;
    
        
    kernel = zeros(N, N);
    for i = 1:N
        for j=i:N
            kernel(i,j) = exp(-params.gammaHeur*sum((X_whole(i,:)-X_whole(j,:)).^2.)); 
            kernel(j,i) = kernel(i,j);
        end
    end

    
    [sortedValues,sortIndex] = sort(kernel(:), 'descend');  
    maxIndex = sortIndex(1:N*kk);  %# Get a linear index into A of the kk largest values
    maxValues = sortedValues(1:N*kk);
    
    
    s=[N,N];
    [I,J]= ind2sub(s,maxIndex); 
    graphInd = cat(2, I(:), J(:));
    graphVal = 2.*(maxValues - 1./params.max_class);
    
end


function gamma = heuristic_gamma(D)
    sigma = sqrt(D/6. - 7./120.);
    gamma = 1./(2*sigma*sigma);
end



