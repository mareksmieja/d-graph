function [maxf, ind] = randIndMax(f)

maxf = max(f);  
ind = find(f == maxf);
maxNum = length(ind);
if maxNum > 1
   ind = ind(randi(maxNum, 1));
end