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

gscatter(F(:,1),F(:,2),y)


%% ensemble binary
X  = F;
y2 = [zeros(14,1);ones(28,1)];
Y = y2;
Mdl = fitcensemble(X,Y,'Method','LogitBoost')
view(Mdl.Trained{1}.CompactRegressionLearner,'Mode','graph');

%% ensemble multiclass

X  = F;
Y = y;
Mdl = fitcensemble(X,Y,'Method','Bag')


%%

cvpart = cvpartition(Y,'holdout',0.3);
Xtrain = X(training(cvpart),:);
Ytrain = Y(training(cvpart),:);
Xtest = X(test(cvpart),:);
Ytest = Y(test(cvpart),:);

Mdl = fitcensemble(Xtrain,Ytrain,'Method','Bag')
Mdl2 = fitcensemble(X,Y,'Method','Bag','Kfold',5)

figure;
plot(loss(Mdl,Xtest,Ytest,'mode','cumulative'));
hold on;
plot(kfoldLoss(Mdl2,'mode','cumulative'),'r.');
hold off;
xlabel('Number of trees');
ylabel('Classification error');
legend('Test','Cross-validation','Location','NE');
