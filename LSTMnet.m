clc;clear all;close all;

load ('DataClass3.mat')
Signals = P.Signal;
Labels  = P.Label;
y       = cellstr(num2str(Labels));
Labels  = categorical(y);
S = num2cell(Signals,2);
%%
S1 = S(Labels=='0');
L1 = Labels(Labels=='0');

S2 = S(Labels=='1');
L2 = Labels(Labels=='1');

S3 = S(Labels=='2');
L3 = Labels(Labels=='2');
%%
[trainInd1,~,testInd1] = dividerand(14,0.9,0.0,0.1);
[trainInd2,~,testInd2] = dividerand(14,0.9,0.0,0.1);
[trainInd3,~,testInd3] = dividerand(14,0.9,0.0,0.1);

XTrain1 = S1(trainInd1);
YTrain1 = L1(trainInd1);

XTrain2 = S2(trainInd2);
YTrain2 = L2(trainInd2);

XTrain3 = S3(trainInd3);
YTrain3 = L3(trainInd3);

XTest1 = S1(testInd1);
YTest1 = L1(testInd1);

XTest2 = S2(testInd2);
YTest2 = L2(testInd2);

XTest3 = S3(testInd3);
YTest3 = L3(testInd3);

XTrain = [XTrain1;XTrain2;XTrain3];
XTest  = [XTest1;XTest2;XTest3];

YTrain = [YTrain1;YTrain2;YTrain3];
YTest  = [YTest1;YTest2;YTest3];
%%
layers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(100,'OutputMode','last')
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer
    ]

options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize', 1101, ...
    'InitialLearnRate', 0.01, ...
    'SequenceLength', 1101, ...
    'GradientThreshold', 1, ...
    'plots','training-progress', ...
    'Verbose',false);

net       = trainNetwork(XTrain,YTrain,layers,options);

trainPred = classify(net,XTrain,'SequenceLength',1101);

plotconfusion(YTrain',trainPred','Training Accuracy')


%% SVM
SVMModel = fitcsvm(X,y)

%% RNN
[X,T] = simpleseries_dataset;
net   = layrecnet(1:2,10);
[Xs,Xi,Ai,Ts] = preparets(net,X,T);
net = train(net,Xs,Ts,Xi,Ai);
view(net)
Y = net(Xs,Xi,Ai);
perf = perform(net,Y,Ts)