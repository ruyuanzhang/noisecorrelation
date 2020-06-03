%% Simulate a pool of voxels on a stimulus-classfication task
% This script corresponds to Figure 5 panels A,B in the manuscript
% This script might run several minutes.

%%
clear all;close all;clc;
nVxs_list = [10, 20, 50, 100, 200, 500];
nTrials = [1000, 1000, 1000, 1000, 2000, 5000]; % how many trials per stim to simulate

Cvxs = [0, 0.01,0.03,0.1,0.3,0.5,0.8,0.99];  % level of noise correlation coeffient
nCvxs = length(Cvxs); 
nSimulations = 10;
stim1 = 80;
stim2 = 100;
wantsave = 0;

%
nNeurons = 180; % number of neurons under a voxel
% define variance of the response variance of the voxels
vxsVarMean = 3;
vxsVarStd = 1;

% compute channel responses of 8 channels for 180 stim
phi = 0: 180 / nNeurons:180; % eight equally spaced orientation channels from 0-180.
phi = phi(2:end);
oriStim = 1:180;
meanNeuronResp = zeros(nNeurons, length(oriStim)); % 

% params tuning curves
alpha = 1;
beta = 19;
gamma = 2;
for i=1:180
    % use the same tuning curves as the real tunings
    meanNeuronResp(:, i) = alpha + beta * exp(gamma * (cos(pi / 90 * (i - phi))-1));
end
%% open parallel
if isempty(gcp) && nSimulations>1
    pobj  = parpool(10);
end
%%
[pCorrect_cTCNC, pCorrect_SFNC] = deal(cell(1, nSimulations));
%for iSimu = 1:nSimulations
parfor iSimu = 1:nSimulations
    iSimu

    [pCorrect_cTCNCtmp, pCorrect_SFNCtmp] = deal(zeros(length(nVxs_list), nCvxs));
    
    for iNvxs = 1:length(nVxs_list)  % loop different voxel size
        
        W = 0.01 * 180/nNeurons * rand(nVxs_list(iNvxs), nNeurons); % for any given pool size, we fixed the conn
        % fix the vaiance matrix
        tau = gamrnd(vxsVarMean^2/vxsVarStd, vxsVarStd/vxsVarMean,[1,nVxs_list(iNvxs)]);
        varMat = sqrt(tau')*sqrt(tau);
        
        % derive mean responses of voxels towards 180 stim
        meanVxsResp = W * meanNeuronResp;  % nVxs_list x nNeurons * nNeurons x stim = nVxs_list X nStim
        % compute signal (tuning) correlation between voxels
        R_SC = corr(meanVxsResp'); % signal correlation
        
        % cTCNC
        R_cTCNC = R_SC;
        
        % Shuffled one
        randOrder = Shuffle(1:nVxs_list(iNvxs));
        [xx,yy] = meshgrid(randOrder);
        ind = sub2ind(size(R_SC), xx, yy);
        R_SFNC = R_SC(ind);
        
        %% tuning compatible noise correlation
        for iCvxs = 1:nCvxs % loop Cvxs
            
            % cTCNC
            tmp = Cvxs(iCvxs)*R_cTCNC;
            tmp(logical(eye(size(tmp)))) = 1;
            Q_cTCvxs = varMat.*tmp;
            
            % generate data and classify
            data1 = mvnrnd(meanVxsResp(:,stim1), Q_cTCvxs, nTrials(iNvxs)); % stim1
            data2 = mvnrnd(meanVxsResp(:,stim2), Q_cTCvxs, nTrials(iNvxs)); % stim2
            sample = [data1(1:nTrials(iNvxs)/2,:); data2(1:nTrials(iNvxs)/2,:)];
            training = [data1(nTrials(iNvxs)/2+1:end,:); data2(nTrials(iNvxs)/2+1:end,:)];
            group = [1*ones(nTrials(iNvxs)/2,1); 2*ones(nTrials(iNvxs)/2,1)];
            [class,~,~,~,classObj] = classify(sample,training,group,'linear');

            good = 0;
            good = good + sum(class(1:nTrials(iNvxs)/2)==1);
            good = good + sum(class(nTrials(iNvxs)/2+1:end)==2);
            pCorrect_cTCNCtmp(iNvxs, iCvxs) = good/nTrials(iNvxs) * 100;
            
            
            %% SFNC
            tmp = Cvxs(iCvxs)*R_SFNC;
            tmp(logical(eye(size(R_SFNC)))) = 1;
            Q_SFNC = varMat.*tmp;
       
            % generate data and classify
            data1 = mvnrnd(meanVxsResp(:,stim1), Q_SFNC, nTrials(iNvxs)); % stim1
            data2 = mvnrnd(meanVxsResp(:,stim2), Q_SFNC, nTrials(iNvxs)); % stim2
            sample = [data1(1:nTrials(iNvxs)/2,:); data2(1:nTrials(iNvxs)/2,:)];
            training = [data1(nTrials(iNvxs)/2+1:end,:); data2(nTrials(iNvxs)/2+1:end,:)];
            group = [1*ones(nTrials(iNvxs)/2,1); 2*ones(nTrials(iNvxs)/2,1)];
            [class,~,~,~,classObj] = classify(sample,training,group,'linear');
            %Mdl = fitclinear(training,group,'FitBias',false,'Learner','logistic');        % fit
            %class = predict(Mdl,sample);% test
            good = 0;
            good = good + sum(class(1:nTrials(iNvxs)/2)==1);
            good = good + sum(class(nTrials(iNvxs)/2+1:end)==2);
            pCorrect_SFNCtmp(iNvxs, iCvxs) = good/nTrials(iNvxs) * 100;
        end 
        
    end
    pCorrect_SFNC{iSimu} = pCorrect_SFNCtmp;
    pCorrect_cTCNC{iSimu} = pCorrect_cTCNCtmp;
end
if ~isempty(gcp)
    delete(gcp);
end

%%
% preprocess the data
pCorrect_cTCNC = cat(3,pCorrect_cTCNC{:});
pCorrect_SFNC = cat(3,pCorrect_SFNC{:});

%%
mn_pCorrect_cTCNC = mean(pCorrect_cTCNC,3);
se_pCorrect_cTCNC = cat(3, zeros(size(mn_pCorrect_cTCNC)), (prctile(pCorrect_cTCNC,97.5,3)-prctile(pCorrect_cTCNC,2.5,3))/2);

mn_pCorrect_SFNC = mean(pCorrect_SFNC,3);
se_pCorrect_SFNC = cat(3, zeros(size(mn_pCorrect_SFNC)), (prctile(pCorrect_SFNC,97.5,3)-prctile(pCorrect_SFNC,2.5,3))/2);

%% plot results
% make legend
legend_label_coeff = cell(1,nCvxs);
for i=1:nCvxs;legend_label_coeff{i} = sprintf('c=%.02f',Cvxs(i)); end
legend_label_nvxs = cell(1,length(nVxs_list));
for i=1:length(nVxs_list);legend_label_nvxs{i} = sprintf('%d',nVxs_list(i)); end

close all;
h1 = cpsfigure(1,2,1);
set(h1,'Position',[0 0 800,300]);
ax(1) = subplot(1,2,1);
[lh,eh] = myplot(nVxs_list,mn_pCorrect_cTCNC',[], '-'); hold on;
c = cool(length(lh));
for i=1:length(lh)
    set(lh(i),'Color',c(i,:));
    if iscell(eh)
        set(eh{i},'Color',c(i,:));
    end
end
xlabel('Number of voxels'); ylabel('Classification accuracy (%)');
title('cTCvxs');
legend(legend_label_coeff);
set(gca, 'XScale','log');

ax(2) = subplot(1,2,2);
[lh,eh] = myplot(nVxs_list,mn_pCorrect_SFNC',[], '-'); hold on;
%c = copper(length(lh));
c = cool(length(lh));
for i=1:length(lh)
    set(lh(i),'Color',c(i,:));
    if iscell(eh)
        set(eh{i},'Color',c(i,:));
    end
end
title('SFNC');
xlabel('Cvxs'); ylabel('Classification accuracy (%)');
set(gca, 'XScale','log');

%
h2 = cpsfigure(1,2,2);
set(h2,'Position',[0 0 800,300]);
ax(1) = subplot(1,2,1);
[lh,eh] = myplot(Cvxs, mn_pCorrect_cTCNC,[], '-'); hold on;
c = parula(length(lh));
for i=1:length(lh)
    set(lh(i),'Color',c(i,:));
    if iscell(eh)
        set(eh{i},'Color',c(i,:));
    end
end
xlabel('Cvxs'); ylabel('Classification accuracy (%)');
title('cTCvxs');
legend(legend_label_nvxs);

ax(2) = subplot(1,2,2);
[lh,eh] = myplot(Cvxs, mn_pCorrect_SFNC,[], '-'); hold on;
c = parula(length(lh));
for i=1:length(lh)
    set(lh(i),'Color',c(i,:));
    if iscell(eh)
        set(eh{i},'Color',c(i,:));
    end
end
title('SFNC');
xlabel('Cvxs'); ylabel('Classification accuracy (%)');

%% Save the result and the figure
if wantsave
    saveas(h1,'vxssimu_classify1.fig');
    saveas(h2,'vxssimu_classify2.fig');
    print(h1,'-dpdf','-painters','-r300','vxssimu_classify1.pdf');
    print(h2,'-dpdf','-painters','-r300','vxssimu_classify2.pdf');
    close all;save('vxssimu_classify.mat');
end
