function bnrlb = numlb2bnrlb(kmlb);

%
%
%
%

N = length(kmlb);

[labs, maxposit, indx] = unique(kmlb);
K = length(labs);

bnrlb  = zeros(N,K);
for i = 1:N
    bnrlb(i, indx(i))= 1;
end



%bnrlb  = zeros(N,K);
%indices = sub2ind(size(bnrlb), (1:N)', kmlb);
%bnrlb(indices) = 1;
%
%count = sum(bnrlb, 1);
%bnrlb = bnrlb(:, count > 0);



