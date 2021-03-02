clc;clear all;close all;

load ('DataClass3.mat')
Signals = P.Signal;
Labels  = P.Label;
y       = cellstr(num2str(Labels));
labels  = categorical(y);
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

%%

X = F;
classifier_name = {'Naive Bayes','Discriminant Analysis','Classification Tree','Nearest Neighbor'};

% Train a naive Bayes model.

classifier{1} = fitcnb(X,y);
% Train a discriminant analysis classifier.

classifier{2} = fitcdiscr(X,y);
% Train a classification decision tree.

classifier{3} = fitctree(X,y);
% Train a k-nearest neighbor classifier.

classifier{4} = fitcknn(X,y);

x1range = min(X(:,1)):.01:max(X(:,1));
x2range = min(X(:,2)):.01:max(X(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];

for i = 1:numel(classifier)
   predictedspecies = predict(classifier{i},XGrid);

   subplot(2,2,i);
   gscatter(xx1(:), xx2(:), predictedspecies,'rgb');

   title(classifier_name{i})
   legend off, axis tight
end

% legend(labels,'Location',[0.35,0.01,0.35,0.05],'Orientation','Horizontal')