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
%%
Xdev = F(:,1) ;
Area = F(:,2);
Class  = [zeros(14,1); ones(14,1);2*ones(14,1)];
tbl = table(Xdev,Area,Class,'VariableNames',{'Xdev','Area','Class'});
lm = fitlm(tbl,'Class~Xdev+Area')

%% Regress plot
x1 = Xdev;
x2 = Area;    % Contains NaN data
y  = Class;
X  = [ones(size(x1)) x1 x2 x1.*x2];
b  = regress(y,X)    % Removes NaN data

scatter3(x1,x2,y,'filled')
hold on
x1fit = min(x1):0.01:max(x1);
x2fit = min(x2):1:max(x2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT + b(4)*X1FIT.*X2FIT;
mesh(X1FIT,X2FIT,YFIT)
xlabel('Xdev')
ylabel('Area')
zlabel('FF')
view(50,10)
hold off

%%  lasso -  find the best predictor
[B,FitInfo] = lasso(F,y,'CV',10,'PredictorNames',{'Xdev','Area'});
idxLambdaMinMSE = FitInfo.IndexMinMSE;
minMSEModelPredictors = FitInfo.PredictorNames(B(:,idxLambdaMinMSE)~=0)
lassoPlot(B,FitInfo,'PlotType','CV');
legend('show') % Show legend


%% ridge

X = [x1 x2];
D = x2fx(X,'interaction');
D(:,1) = []; % No constant term
k = 0:1e-5:5e-3;
b = ridge(y,D,k);

figure
plot(k,b,'LineWidth',2)
ylim([-100 100])
grid on
xlabel('Ridge Parameter')
ylabel('Standardized Coefficient')
title('{\bf Ridge Trace}')
legend('x1','x2','x1x2')
