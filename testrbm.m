%%data = csvread('train2INTRmovetrainingfilex')
%%labels = csvread('train2INTRmovetrainingfiley')
%%testdata = csvread('test2INTRmovetrainingfilex')
%%testlabels = csvread('test2INTRmovetrainingfiley')

load('matlab-2000f-6000s-1000t.mat');


%% Train RBM for classification
%train rbm with 100 hidden units
opts.eta=0.08;
opts.momentum=0.1;
opts.batchsize=50;
opts.penalty=1e-9;

m=rbmFit(data,1000,labels,opts);
yhat=rbmPredict(m,testdata);

%print error
fprintf('Classification error using RBM with 100 hiddens is %f\n', ...
    sum(yhat~=testlabels)/length(yhat));

%visualize weights
figure(1)
visualize(m.W);
title('learned weights');

%visualize the mislabeled cases. Note the transpose. Visualize assumes DxN
%as is the case for weights
figure(2)
visualize(testdata(yhat~=testlabels,:)');
title('classification mistakes for RBM with 100 hiddens');
drawnow;
