function [DataInstances, Header,Constraints, ClassLabeled, ClassLabels]=loadData( datafilename, constraintfilename )

    fid = fopen(datafilename);
    temp = fscanf(fid, '%d %d %d');

    NoOfObjs = temp(1);
    NoOfFeatures = temp(2);
    ClassLabeled = temp(3);

    ClassLabels = zeros(NoOfObjs, 1);
    
    %size(NoOfObjs);
    
    %load header info%
    Header = zeros(NoOfFeatures, 1); % 1 = numerical, 0 = categorical
    temp = char(textscan(fid, '%c', NoOfFeatures));
    for i = 1:NoOfFeatures
        if temp(i)=='N' || temp(i)=='n'
            Header(i) = true;
        else 
            Header(i) = false;
        end
    end

    % load data instances %
    DataInstances = zeros(NoOfObjs, NoOfFeatures );
    for i = 1:NoOfObjs
        textscan(fid, '%d',1); % discard line id
        DataInstances(i,:) = cell2mat(textscan(fid, '%f', NoOfFeatures));
        if ClassLabeled==true
            ClassLabels(i) = cell2mat(textscan(fid, '%d', 1));
        end
    end
    
    DataInstances = normalizeRange(DataInstances);
    fclose(fid);
    fprintf('%d Data Instances loaded\n', NoOfObjs)
    fprintf('%d Features\n', NoOfFeatures)
    
    % one pass to count how many constraints 
    count = 0;
    fid = fopen(constraintfilename);
    while ~feof(fid)
        [temp, succInLine] = fscanf(fid, '%d %d %d', 3);
        if succInLine==3
            count = count+1;
        end
    end

    fprintf('%d Constraints loaded\n', count)
    Constraints = zeros(count * 2, 3);
    fseek(fid,0, 'bof');
    for i=1:count
        Constraints(i*2 - 1,:) = fscanf(fid, '%d %d %d', 3) + 1;
        Constraints(i*2,:) = [ Constraints(i*2-1,2) Constraints(i*2-1,1) Constraints(i*2-1,3) ];
    end
    fclose(fid);
end

function xn = normalizeToStd1(x)

%normalize each feature so that the stddev equals to 1 
%and the mean is 0
n = size(x, 1);
p = size(x, 2);
xn = zeros( n, p );
xmean = zeros(p,1);
xscale = zeros(p,1);
for j=1:p
    xmean(j) = sum( x(:,j) )/n;
    xn(:,j) = x(:,j) - xmean(j);
    xscale(j) = std( xn(:,j) ); %sqrt(sum(xn(:,j).^2));
    xn(:,j) = xn(:,j) / xscale(j);
end
end


function xn = normalizeRange(x)

n = size(x, 1);
p = size(x, 2);
xn = zeros( n, p );

for j=1:p
    xmax = max( x(:,j) );
    xmin = min( x(:,j) );
    xn(:,j) = (x(:,j) - xmin) / (xmax-xmin);
end
end
