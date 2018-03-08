function kernel = computeKernel(X_whole, params)
    if isequal(params.kernel, 'RBF')
        [N, ~] = size(X_whole);
        kernel = zeros(N, N);
        for i = 1:N
            for j=i:N
                kernel(i,j) = exp(-params.gamma*sum((X_whole(i,:)-X_whole(j,:)).^2.)); 
                kernel(j,i) = kernel(i,j);
            end
        end
    elseif isequal(params.kernel, 'Tanimoto')
        [N, ~] = size(X_whole);
        kernel = zeros(N, N);
        for i = 1:N
            for j=i:N
                kernel(i,j) = (X_whole(i,:).*X_whole(j,:)) / (X_whole(i,:).*X_whole(i,:) + X_whole(j,:).*X_whole(j,:) - X_whole(i,:).*X_whole(j,:));
                kernel(j,i) = kernel(i,j);
            end
        end
    else
        kernel = X_whole;
    end
end