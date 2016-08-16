%data = csvread('train2INTRmovetrainingfilex')
%labels = csvread('train2INTRmovetrainingfiley')
%testdata = csvread('test2INTRmovetrainingfilex')
%testlabels = csvread('test2INTRmovetrainingfiley')

load('matlab-2000f-6000s-1000t.mat');
%% Train RBM for classification
%train rbm with 100 hidden units
opts.eta=0.1;
opts.momentum=0.5;
opts.batchsize=100;
opts.penalty=2e-4;

opts1.eta=0.08;
opts1.momentum=0.1;
opts1.batchsize=50;
opts1.penalty=1e-10;

m=dbnFit(data,[1000 2000],labels,opts,opts1);
yhat=dbnPredict(m,testdata);

%print error
fprintf('Classification error using RBM with 100 hiddens is %f\n', ...
    sum(yhat~=testlabels)/length(yhat));

%visualize the mislabeled cases. Note the transpose. Visualize assumes DxN
%as is the case for weights
figure(1)
visualize(yhat~=testlabels);
figure(2)
visualize(testdata(yhat~=testlabels,:)');
title('classification mistakes for RBM with 100 hiddens');
drawnow;
