function [minf, ind] = randIndMin(f)

minf = min(f);  
ind = find(f == minf);
maxNum = length(ind);
if maxNum > 1
   ind = ind(randi(maxNum, 1));
end