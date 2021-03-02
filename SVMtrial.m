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


%% 2 class SVM
X  = F;
y2 = [zeros(14,1);ones(28,1)];
SVMModel   = fitcsvm(X,y2)
classOrder = SVMModel.ClassNames
sv         = SVMModel.SupportVectors;
figure
gscatter(X(:,1),X(:,2),y2)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
legend('Baseline','Forcefield','Support Vector')
hold off

%% multi class SVM - error correcting output codes

X = F;
Y = y;
t              = templateSVM('Standardize',true,'SaveSupportVectors',true);
predictorNames = {'Xdeviation','Area'};
responseName   = 'ReachMovement';
classNames     = {'0','1','2'}; % Specify class order
Mdl            = fitcecoc(X,Y,'Learners',t,'ResponseName',responseName,...
    'PredictorNames',predictorNames,'ClassNames',classNames)
Mdl.ClassNames
CodingMat = Mdl.CodingMatrix
Mdl.BinaryLearners{1} 

L = size(Mdl.CodingMatrix,2); % Number of SVMs
sv = cell(L,1); % Preallocate for support vector indices
for j = 1:L
    SVM = Mdl.BinaryLearners{j};
    sv{j} = SVM.SupportVectors;
    sv{j} = sv{j}.*SVM.Sigma + SVM.Mu;
end

figure
gscatter(X(:,1),X(:,2),Y);
hold on
markers = {'ko','ro','bo'}; % Should be of length L
for j = 1:L
    svs = sv{j};
    plot(svs(:,1),svs(:,2),markers{j},...
        'MarkerSize',10 + (j - 1)*3);
end
title('ReachMovement -- ECOC Support Vectors')
xlabel(predictorNames{1})
ylabel(predictorNames{2})
legend([classNames,{'Support vectors - SVM 1',...
    'Support vectors - SVM 2','Support vectors - SVM 3'}],...
    'Location','Best')
hold off

%%
CVMdl    = crossval(Mdl);
genError = kfoldLoss(CVMdl)