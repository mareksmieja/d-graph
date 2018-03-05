% This function reports the Pairwise-F1 accuracy of clustering

function [randIndex, PWF1] = reportAccuracy( ClassLabels, assignedCentre ) 
    NoOfObjs = size(ClassLabels,1);
    sum = 0;
    p1 = 0;
    p2 = 0;
    p3 = 0;
    
    for i = 1:NoOfObjs
        for j = i+1:NoOfObjs
            
            if (  (assignedCentre(i)==assignedCentre(j)) ==  (ClassLabels(i) == ClassLabels(j)) )
                sum=sum+1;
            end
            if (  (assignedCentre(i)==assignedCentre(j)) )
                p2=p2+1;
            end            
            if (  (assignedCentre(i)==assignedCentre(j)) &&  (ClassLabels(i) == ClassLabels(j)) )
                p1=p1+1;
            end
            if (  ClassLabels(i) == ClassLabels(j) )
                p3=p3+1;
            end            
        end
    end
    
    precision = p1/p2;
    recall = p1/p3;
    
    PWF1 = 2*precision * recall/(precision + recall);
    randIndex =  sum/( NoOfObjs * ( NoOfObjs-1)/2);
    
end