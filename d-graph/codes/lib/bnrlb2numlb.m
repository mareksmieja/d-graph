function numlb = bnrlb2numlb(bnrlb)

[N,K] = size(bnrlb);

numlb = zeros(N,1);
for i = 1:N
    numlb(i,1) = find(bnrlb(i,:) ==1);
end