function [graphInd, graphVal] = findGraphTopKernelNN(X_whole, params)
    [N,D] = size(X_whole);
    kk= params.knn;
    
    kernel = X_whole;
    
    [sortedValues,sortIndex] = sort(kernel(:), 'descend');  
    maxIndex = sortIndex(1:N*kk);  %# Get a linear index into A of the kk largest values
    maxValues = sortedValues(1:N*kk);
    
    
    s=[N,N];
    [I,J]= ind2sub(s,maxIndex); 
    graphInd = cat(2, I(:), J(:));
    graphVal = 2.*(maxValues - 1./params.max_class);
    
    
end



