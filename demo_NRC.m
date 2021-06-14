% =========================================================================
%   Lei Zhang, Meng Yang, and Xiangchu Feng,
%   "Sparse Representation or Collaborative Representation: Which Helps Face
%    Recognition?" in ICCV 2011.
%
%
% Written by Meng Yang @ COMP HK-PolyU
% July, 2011.
% =========================================================================
                      
close all;
clear all;
clc;

% -------------------------------------------------------------------------
% parameter setting
par.nClass        =   100;                 % the number of classes in the subset of AR database
par.nDim          =   300;                 % the eigenfaces dimension

option.iter = 20;
option.threshold = 0.001;
option.rho = 0.01;

%--------------------------------------------------------------------------
%data loading (here we use the AR dataset as an example)
load(['AR_DAT']);
Tr_DAT   =   double(NewTrain_DAT(:,trainlabels<=par.nClass));
trls     =   trainlabels(trainlabels<=par.nClass);
Tt_DAT   =   double(NewTest_DAT(:,testlabels<=par.nClass));
ttls     =   testlabels(testlabels<=par.nClass);
clear NewTest_DAT NewTrain_DAT testlabels trainlabels

%--------------------------------------------------------------------------
%eigenface extracting
[disc_set,disc_value,Mean_Image]  =  Eigenface_f(Tr_DAT,par.nDim);
tr_dat  =  disc_set'*Tr_DAT;
tt_dat  =  disc_set'*Tt_DAT;
tr_dat  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [par.nDim,1]) );
tt_dat  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [par.nDim,1]) );

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

