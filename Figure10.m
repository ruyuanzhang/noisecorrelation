%% Fig.10, using 3-units model to illustrate winner-take-all principle in the linear classifer.
% This script correspond to Figure 10

clear all;close all;clc
% Simulate two random variables
nTrainTrials = 10000;
nTestTrials = 10000;

mean1 = [1, 0, 0.5];  % the mean response of three voxels to stimulus 1
mean2 = [0, 1, -0.5]; % the mean response of three voxels to stimulus 2

% get the covariance
variance = 1;
% noise correlation matrix
sigma = [1 0.5 0; ...
        0.5 1 -0.5; ...
        0, -0.5 1] * variance;

train1 = mvnrnd(mean1, sigma, nTrainTrials);
train2 = mvnrnd(mean2, sigma, nTrainTrials);

test1 = mvnrnd(mean1, sigma, nTestTrials);
test2 = mvnrnd(mean2, sigma, nTestTrials);

% control as no NC
sigma_noNC = [1 0 0;
    0 1 0;
    0 0 1] * variance;

train_noNC1 = mvnrnd(mean1, sigma_noNC, nTrainTrials);
train_noNC2 = mvnrnd(mean2, sigma_noNC, nTrainTrials);
test_noNC1 = mvnrnd(mean1, sigma_noNC, nTestTrials);
test_noNC2 = mvnrnd(mean2, sigma_noNC, nTestTrials);


%% plot test1 test2
close all;
h = cpsfigure(2,2);
set(h, 'Position', [0 0 650 600]);
% 
ax(1) = subplot(2,2,1);  % x vs. y
myplot(test1(:,1)', test1(:,2)', [], '.'); hold on;  
myplot(test2(:,1)', test2(:,2)', [],'.'); hold on;
xlabel('Response of voxel X');ylabel('Response of voxel Y'); legend({'Stim1','Stim2'})

ax(2) = subplot(2,2,2);  % y vs. z
myplot(test1(:,2)', test1(:,3)', [], '.'); hold on;  
myplot(test2(:,2)', test2(:,3)', [], '.'); hold on;
xlabel('Response of voxel Y');ylabel('Response of voxel Z');

ax(3) = subplot(2,2,3);  % x vs. z
myplot(test1(:,1)', test1(:,3)', [], '.'); hold on;  
myplot(test2(:,1)', test2(:,3)', [],  '.'); hold on;
xlabel('Response of voxel X');ylabel('Response of voxel Z');

%% classification 
% only classify x,y
training = [train1(:,1:2); train2(:,1:2)];
test = [test1(:,1:2);test2(:,1:2)];
group = [ones(nTrainTrials,1); 2*ones(nTrainTrials,1)];

% fisher linear
class = classify(test,training,group,'linear');

% logistic
%Mdl = fitclinear(training,group,'FitBias',false,'Learner','logistic');        % fit
%class = predict(Mdl,test);% test

pCorr_xy = sum(class==group)/length(class) * 100;

% only classify y,z
training = [train1(:,2:3); train2(:,2:3)];
test = [test1(:,2:3);test2(:,2:3)];
group = [ones(nTrainTrials,1); 2*ones(nTrainTrials,1)];

class = classify(test,training,group,'linear');

%Mdl = fitclinear(training,group,'FitBias',false,'Learner','logistic');        % fit
%class = predict(Mdl,test);% test

pCorr_yz = sum(class==group)/length(class) * 100;

% only classify x,z
training = [train1(:,[1 3]); train2(:,[1 3])];
test = [test1(:,[1 3]);test2(:,[1 3])];
group = [ones(nTrainTrials,1); 2*ones(nTrainTrials,1)];
class = classify(test,training,group,'linear');
pCorr_xz = sum(class==group)/length(class) * 100;

% classify using all
training = [train1; train2];
test = [test1;test2];
group = [ones(nTrainTrials,1); 2*ones(nTrainTrials,1)];
class = classify(test,training,group,'linear');
pCorr_xyz = sum(class==group)/length(class) * 100;

% classify using all but no NC between three units
training = [train_noNC1; train_noNC2];
test = [test_noNC1; test_noNC2];
group = [ones(nTrainTrials,1); 2*ones(nTrainTrials,1)];
class = classify(test,training,group,'linear');
pCorr_xyz_noNC = sum(class==group)/length(class) * 100;

% plot the bar figure
ax(4) = subplot(2, 2, 4);
c = gray(5);
Hb(1) = mybar(1,pCorr_xy);
Hb(2) = mybar(2,pCorr_yz);
Hb(3) = mybar(3,pCorr_xz);
Hb(4) = mybar(4,pCorr_xyz);
Hb(5) = mybar(5,pCorr_xyz_noNC);
for i=1:5,set(Hb(i),'FaceColor',c(i,:));end
ylabel('Classification accuracy (%)');
ylim([60, 100]);xlim([0, 6]);
set(gca, 'XTickLabels',{});
legend(Hb, {'Classify from X&Y','Classify from Y&Z','Classify from X&Z','Classify from X,Y&Z','Classify from X,Y&Z without NC'});

%% save it
print(h,'-dpdf','-painters','-r300','Figure10raw.pdf');