close all;
clear all;
clc;

name = 'YaleB_32x32'
load (name);
fea = double(fea);
sele_num = 32;
Eigen_NUM=300;

%ADMM option
option.iter = 20;
option.threshold = 0.001;
option.rho = 0.01;


nnClass = length(unique(gnd));  % The number of classes;
num_Class = [];
for i = 1:nnClass
  num_Class = [num_Class length(find(gnd==i))]; %The number of samples of each class
end
%%------------------select training samples and test samples--------------%%
Train_Ma  = [];
Train_Lab = [];
Test_Ma   = [];
Test_Lab  = [];
for j = 1:nnClass    
    idx      = find(gnd==j);
    randIdx  = randperm(num_Class(j));
    Train_Ma = [Train_Ma; fea(idx(randIdx(1:sele_num)),:)];            % select select_num samples per class for training
    Train_Lab= [Train_Lab;gnd(idx(randIdx(1:sele_num)))];
    Test_Ma  = [Test_Ma;fea(idx(randIdx(sele_num+1:num_Class(j))),:)];  % select remaining samples per class for test
    Test_Lab = [Test_Lab;gnd(idx(randIdx(sele_num+1:num_Class(j))))];
end
Train_Ma = Train_Ma';                       % transform to a sample per column
Train_Ma = Train_Ma./repmat(sqrt(sum(Train_Ma.^2)),[size(Train_Ma,1) 1]);
Test_Ma  = Test_Ma';
Test_Ma  = Test_Ma./repmat(sqrt(sum(Test_Ma.^2)),[size(Test_Ma,1) 1]); 

Tr_DAT = Train_Ma;
Tt_DAT = Test_Ma;
trls = Train_Lab';
ttls = Test_Lab';


%eigenface extracting
[disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,Eigen_NUM);
tr_dat  =  disc_set'*Tr_DAT;
tt_dat  =  disc_set'*Tt_DAT;
tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [Eigen_NUM,1]) );
tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [Eigen_NUM,1]) );

%-------------------------------------------------------------------------

ID = [];
[z_t]    = ADMM(tr_dat,tt_dat,option);
for indTest = 1:size(tt_dat,2)
%     indTest
    [id]= NRC(tr_dat,z_t(:,indTest),tt_dat(:,indTest),trls);
    ID      =   [ID id];
end
cornum      =   sum(ID==ttls);
Rec         =   [cornum/length(ttls)]; % recognition rate
fprintf(['recogniton rate is ' num2str(Rec) '\n']);