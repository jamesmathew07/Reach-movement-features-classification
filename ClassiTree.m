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


%% Class tree
X  = F;
y2 = [zeros(14,1);ones(28,1)];
MdlDefault   = fitctree(X,y,'CrossVal','on')
view(MdlDefault.Trained{1},'Mode','graph')
classErrorDefault = kfoldLoss(MdlDefault)
classError7 = kfoldLoss(Mdl7)

% Mdl7 = fitctree(X,y,'MaxNumSplits',5,'CrossVal','on');
% view(Mdl7.Trained{1},'Mode','graph')
