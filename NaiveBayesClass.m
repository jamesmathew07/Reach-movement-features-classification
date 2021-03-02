clc;clear all;close all;

load ('DataClass3.mat')
Signals = P.Signal;
Labels  = P.Label;
y       = cellstr(num2str(Labels));
Labels  = categorical(y);
S = num2cell(Signals,2);

%%
% features
Signals2 = Signals-0.12;
for i = 1:28
    F(i,1) = min(Signals2(i,:));
    F(i,2) = sum(abs(Signals2(i,:)));
end
for i=29:42
    F(i,1) = max(Signals2(i,:));
    F(i,2) = sum(abs(Signals2(i,:)));
end

scatter(F(:,1),F(:,2))


%% Naive bayes
tabulate(y)
X  = F;
Mdl  = fitcnb(X,y)
Mdl.Mu

%% plot
Y = y;
figure
gscatter(X(:,1),X(:,2),Y);
h      = gca;
cxlim  = h.XLim;
cylim  = h.YLim;
hold on
Params = cell2mat(Mdl.DistributionParameters); 
Mu     = Params(2*(1:3)-1,1:2); % Extract the means
Sigma = zeros(2,2,3);
for j = 1:3
    Sigma(:,:,j) = diag(Params(2*j,:)).^2; % Create diagonal covariance matrix
    xlim         = Mu(j,1) + 4*[1 -1]*sqrt(Sigma(1,1,j));
    ylim         = Mu(j,2) + 4*[1 -1]*sqrt(Sigma(2,2,j));
    ezcontour(@(x1,x2)mvnpdf([x1,x2],Mu(j,:),Sigma(:,:,j)),[xlim ylim])
        % Draw contours for the multivariate normal distributions 
end
h.XLim = cxlim;
h.YLim = cylim;
title('Naive Bayes Classifier -- Reach Movements')
xlabel('X deviation')
ylabel('Area')
hold off

%% Without prior probability  - 2 class
tabulate(y)
y2 = [zeros(14,1);ones(28,1)];
Y = y2;
Mdl  = fitcnb(X,Y)
figure
gscatter(X(:,1),X(:,2),Y);
h      = gca;
cxlim  = h.XLim;
cylim  = h.YLim;
hold on
Params = cell2mat(Mdl.DistributionParameters); 
Mu     = Params(2*(1:2)-1,1:2); % Extract the means
Sigma = zeros(2,2,3);
for j = 1:2
    Sigma(:,:,j) = diag(Params(2*j,:)).^2; % Create diagonal covariance matrix
    xlim         = Mu(j,1) + 4*[1 -1]*sqrt(Sigma(1,1,j));
    ylim         = Mu(j,2) + 4*[1 -1]*sqrt(Sigma(2,2,j));
    ezcontour(@(x1,x2)mvnpdf([x1,x2],Mu(j,:),Sigma(:,:,j)),[xlim ylim])
        % Draw contours for the multivariate normal distributions 
end
h.XLim = cxlim;
h.YLim = cylim;
title('Naive Bayes Classifier -- Reach Movements')
xlabel('X deviation')
ylabel('Area')

%% with prior probability

prior = [0.3 0.7];
Mdl = fitcnb(X,Y,'Prior',prior)

%% compare classifiers

Md1  = fitcnb(X,y,'CrossVal','on');
t = templateNaiveBayes();
Md2 = fitcecoc(X,y,'CrossVal','on','Learners',t);

classErr1 = kfoldLoss(Md1,'LossFun','ClassifErr')

classErr2 = kfoldLoss(Md2,'LossFun','ClassifErr')
