
%% Extracting the data
clear all
data_flag=4; %1 for train, 2 for validation, 3 for test, 4 for no split (takes all the data)

mainDir='C:\Users\user\Google Drive\DataHack\Rocket challenge\';
csvSetTable = readtable([mainDir, 'train_processed', '.csv']);
titles=csvSetTable.Properties.VariableNames;
data = table2array(csvSetTable(:,1:181));
labels=data(:,181);
ind(:,1)=1:size(data,1);

csvSetTable_unlabeled = readtable([mainDir, 'test_processed', '.csv']);
data_unlabel = table2array(csvSetTable_unlabeled(:,1:180));

% checking the distribution of classes in the labeled data
for i=1:max(labels)
class_distr(i)=sum(labels==i)./length(labels);
end

%% loading features from csv
% csvFeatTable = readtable([mainDir, 'final_train_Rafael', '.csv']);
% titlesFeat=csvSetTable.Properties.VariableNames;
% data_Feat{:} = table2array(csvFeatTable(:,1:216));
% % labels is the labels of these data
%% split data to train and test
% ind_perm(:,1)=randperm(length(ind));
% ind_train=ind_perm(1:ceil(0.8*length(ind_perm)));
% ind_valid=ind_perm(length(ind_train)+1:ceil(0.9*length(ind_perm)));
% ind_test=ind_perm(length(ind_train)+length(ind_valid)+1:length(ind_perm));

% csvwrite('ind_train.csv',ind_train)
% csvwrite('ind_valid.csv',ind_valid)
% csvwrite('ind_test.csv',ind_test)
%% load splitted data (not used)
ind_train(:,1)=csvread('ind_train.csv');
ind_valid(:,1)=csvread('ind_valid.csv');
ind_test(:,1)=csvread('ind_test.csv');
%%


switch data_flag
    case 1
        data_used=data(ind_train,:);
        labels_used=labels(ind_train,:);
    case 2
        data_used=data(ind_valid,:);
        labels_used=labels(ind_valid,:);
    case 3
        data_used=data(ind_test,:);
        labels_used=labels(ind_test,:);
    case 4   
        data_used=data;
        labels_used=labels;
end

%% calculating the weighted velocities from labeled data

locX=data_used(:,[1:6:end-1]);
locY=data_used(:,[2:6:end-1]);
locZ=data_used(:,[3:6:end-1]);
velX=data_used(:,[4:6:end-1]);
velY=data_used(:,[5:6:end-1]);
velZ=data_used(:,[6:6:end-1]);
%% Features
%location weighted vector
locTotal=sqrt(locX.^2+locY.^2+locZ.^2);
locTotal_short=locTotal(:,1:10);
%velocity weighted vector
velTotal=sqrt(velX.^2+velY.^2+velZ.^2);
velTotal_short=velTotal(:,1:10);
%velocity mean
velMean=nanmean(velTotal,2);
%acceleration weighted vector
accTotal=diff(velTotal,1,2);
%vel fft
velfft=[];velfftM=[];velfftlog=[];
sf=2;
y=velTotal(:,1:10)';
[velfft,f] = fn_fftAmp(y,sf,0,0);
velfftlog=log(velfft);
velfftlog=velfftlog';
velfft=velfft';
for indf=1:25
velfftM(indf,:)=nanmean(velfftlog(labels_used==indf,:),1);
end
flog=log(f);
%% vel-fft-PCA
train_labels=[]; X=[]; Xlab=[]; X_perm=[]; ind_X=[];mappedX=[];labelsfft=[];velfft2show=[];Uf=[];classes2show=[];Sf=[]; Vf=[];
[Uf,Sf,Vf] = svds(velfftlog,2);
clusters=indf;
figure
gscatter(Uf(:,1), Uf(:,2), labels_used);

%% ordered data (gets same result as previous)
% classes2show=[1:25];
% for indf=classes2show
% velfft2show=[velfft2show;velfftlog(labels_used==indf,:)];
% labelsfft=[labelsfft; labels_used(labels_used==indf)];
% end
% % [Uf,Sf,Vf] = svds(velfft2show',2);
% [Uf_ord,Sf_ord,Vf_ord] = svds(velfft2show,2);
% clusters=indf;
% figure
% gscatter(Uf_ord(:,1), Uf_ord(:,2), labelsfft);

%% permutated data (gets same result as previous)
% X=velfft2show; % velConcatenate_all;
% Xlab=labelsfft;
% ind_X(:,1) = randperm(size(X,1));
% X_perm = X(ind_X,:);
% train_labels = Xlab(ind_X);
% 
% [Uf_perm,Sf_perm,Vf_perm] = svds(X_perm,2);
% clusters=indf;
% figure
% gscatter(Uf_perm(:,1), Uf_perm(:,2), train_labels);

%% loc fft
locfft=[];locfftM=[];locfftlog=[];
sf=2;
y=locTotal(:,1:10)';
[locfft,f] = fn_fftAmp(y,sf,0,0);
locfftlog=log(locfft);
locfftlog=locfftlog';
locfft=locfft';
for indf=1:25
locfftM(indf,:)=nanmean(locfftlog(labels_used==indf,:),1);
end
flog=log(f);
% loc-fft-PCA
train_labels=[]; X=[]; Xlab=[]; X_perm=[]; ind_X=[];mappedX=[];labelsfft=[];locfft2show=[];Uf_loc=[];classes2show=[];Sf=[]; Vf=[];
[Uf_loc,Sf,Vf] = svds(locfftlog,2);
clusters=indf;
figure
gscatter(Uf_loc(:,1), Uf_loc(:,2), labels_used);

%%
% obj  = gmdistribution.fit(Uf,clusters, 'Replicates',500); %fitting the scattered dots into 2 gaussians.
% idx = cluster(obj,Uf); %returns a vector same length as U, with the group number for each entry according to the clustering

% pcaresults=[Uf, labels_used];
% csvwrite('PCA2dim.csv',pcaresults);

% figure
% hold on
% scatter(Uf(idx==1,1),Uf(idx==1,2),3,'b','filled');
% scatter(Uf(idx==2,1),Uf(idx==2,2),3,'r','filled');

% Plot single-sided amplitude spectrum.
% figure
% plot(log(f),log(nanmean(velfft,2))) 
% title('Single-Sided Amplitude Spectrum')
% xlabel('Frequency [Hz]')
% ylabel('Power spectral density [mV^2/Hz]')
%% visualisation of velocity and acceleration
figure
% plot([0.5:0.5:14.5],accTotal(labels_used==7,:))
plot([0:0.5:14.5],velTotal(labels_used==21,:))
%% cut long vectors
vec20=not(isnan(velTotal(:,20))); %the samples with length of at least 20
vec30=not(isnan(velTotal(:,30))); %the samples with length of at least 30
velConcatenate=[velTotal_short; velTotal(vec20==1,11:20); velTotal(vec30==1,21:30)];
labelsConcatenate=[labels_used;labels_used(vec20==1); labels_used(vec30==1)];

%% calculating the weighted velocities from unlabeled data

locX_unlabel=data_unlabel(:,[1:6:end-1]);
locY_unlabel=data_unlabel(:,[2:6:end-1]);
locZ_unlabel=data_unlabel(:,[3:6:end-1]);
velX_unlabel=data_unlabel(:,[4:6:end-1]);
velY_unlabel=data_unlabel(:,[5:6:end-1]);
velZ_unlabel=data_unlabel(:,[6:6:end]);

locTotal_unlabel=sqrt(locX_unlabel.^2+locY_unlabel.^2+locZ_unlabel.^2);
velTotal_unlabel=sqrt(velX_unlabel.^2+velY_unlabel.^2+velZ_unlabel.^2);
locTotal_short_unlabel=locTotal_unlabel(:,1:10);
velTotal_short_unlabel=velTotal_unlabel(:,1:10);

%% cut long vectors from unlabeled data
vec20_unlabel=not(isnan(velTotal_unlabel(:,20))); %the samples with length of at least 20
vec30_unlabel=not(isnan(velTotal_unlabel(:,30))); %the samples with length of at least 30
velConcatenate_unlabel=[velTotal_short_unlabel; velTotal_unlabel(vec20==1,11:20); velTotal_unlabel(vec30==1,21:30)];

%% make one matrix with labeled and unlabeled concatenated traces

velConcatenate_all=[velConcatenate; velConcatenate_unlabel];

%% vel fft unlabled data
velfft_unlabel=[]; velfftlog_unlabel=[];y=[];
sf=2;
y=velTotal_unlabel(:,1:10)';
[velfft_unlabel,f] = fn_fftAmp(y,sf,0,0);
velfftlog_unlabel=log(velfft_unlabel);
velfftlog_unlabel=velfftlog_unlabel';
velfft_unlabel=velfft_unlabel';
labels_used_unlabel(:,1)=26.*ones(1,size(velTotal_unlabel,1));
flog=log(f);
% vel-fft-PCA
train_labels=[]; X=[]; Xlab=[]; X_perm=[]; ind_X=[];mappedX=[];labelsfft=[];velfft2show_unlabel=[];Uf_unlabel=[];classes2show=[];Sf=[]; Vf=[];
[Uf_unlabel,Sf,Vf] = svds(velfftlog_unlabel,2);
clusters=indf;
figure
gscatter(Uf_unlabel(:,1), Uf_unlabel(:,2), labels_used_unlabel);

pcaresults_unlabel=[Uf_unlabel, labels_used_unlabel];
csvwrite('PCA2dimUnlabeled.csv',pcaresults_unlabel);
%% dimension reduction using t-SNE algorithm
% train_labels=[]; X=[]; Xlab=[]; X_perm=[]; mappedX=[]; data2reduce=[];labels2reduce=[];
% data2reduce=velfftlog; % velTotal_short; %velfftlog; %velConcatenate; velfftlog; velTotal_short
% labels2reduce=labels_used; %labelsConcatenate;labels_used
% 
% X=data2reduce; % velConcatenate_all;
% Xlab=labels2reduce;
% % ind_X = randperm(size(X,1));
% % X_perm = X(ind_X(1:1000),:);
% % train_labels = Xlab(ind_X(1:1000));
% X_perm=X(1:5000,:);
% train_labels=Xlab(1:5000,:);
% % Set parameters
% no_dims = 2;
% initial_dims = 2;
% perplexity = 100;
% % Run t?SNE
% mappedX = tsne(X_perm,'NumDimensions',no_dims, 'NumPCAComponents', initial_dims,'Perplexity', perplexity);% Plot results
% figure
% gscatter(mappedX(:,1), mappedX(:,2), train_labels);
% 
% %% PCA
% [U,S,V] = svds(X_perm,2);
% clusters=2;
% obj  = gmdistribution.fit(U,clusters, 'Replicates',500); %fitting the scattered dots into 2 gaussians.
% idx = cluster(obj,U); %returns a vector same length as U, with the group number for each entry according to the clustering
% 
% figure
% gscatter(U(:,1), U(:,2), train_labels);
% 
% figure
% hold on
% scatter(U(idx==1,1),U(idx==1,2),3,'b','filled');
% scatter(U(idx==2,1),U(idx==2,2),3,'r','filled');
% % scatter(U(idx==3,1),U(idx==3,2),3,'g','filled');
% % scatter(U(idx==4,1),U(idx==4,2),5,'y','filled');
% hold off


