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


%% LDA
X  = F;
Mdl  = fitcdiscr(X,y)
Mdl.Mu

%% plot

h1 = gscatter(X(:,1),X(:,2),y,'krb','ov^',[],'off');
h1(1).LineWidth = 2;
h1(2).LineWidth = 2;
h1(3).LineWidth = 2;
legend('base','CCW','CW','Location','best')
hold on
Mdl.ClassNames([2 3])
K = Mdl.Coeffs(2,3).Const;
L = Mdl.Coeffs(2,3).Linear;
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f, [-0.15,0.15,0,35]);
h2.Color = 'm';
h2.LineWidth = 2;

hold on;
Mdl.ClassNames([1 2])
K = Mdl.Coeffs(1,2).Const;
L = Mdl.Coeffs(1,2).Linear;
f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
h2 = ezplot(f, [-0.15,0.15,0,35]);
h2.Color = 'g';
h2.LineWidth = 2;

%% Regularization : optimum predictors  : Cross validated regularization
% Regularization is the process of finding a small set of predictors that yield \
% an effective predictive model
obj = fitcdiscr(X,y,'SaveMemory','on','FillCoeffs','off');
[err,gamma,delta,numpred] = cvshrink(obj,...
    'NumGamma',9,'NumDelta',9,'Verbose',1);
plot(err,numpred,'k*');xlim([-0.2, 0.8]);ylim([-1, 3])
xlabel('Error rate');
ylabel('Number of predictors'); title ('Regularization')