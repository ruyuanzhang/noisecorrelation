%% Simulate a pool of voxels on a stimulus-estimation task
% This script corresponds to Figure 5 panels C,D in the manuscript
% This script might run several minutes.

%% 
clear all;close all;clc;

nVxs_list = [10, 20, 50, 100, 200, 500];
Cvxs = [0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8, 0.99];  

nNeurons = 180;
nTrials = 1000; % how many trials per stim to simulate

% set noise levels for voxels
vxsVarMean = 3;
vxsVarStd = 1;
wantsave = 1;

nSimulations = 10;
nCvxs = length(Cvxs); 
%% compute channal responses of 8 channels for 180 stim
phi = 0: 180 / nNeurons:180; % eight equally spaced orientation channels from 0-180.
phi = phi(2:end);
oriStim = 1:180;
meanNeuronResp = zeros(nNeurons, length(oriStim)); % 

% Params tuning curves
alpha = 1;
beta = 19;
gamma = 2;
for i=1:180
    % use the same tuning curves as the real tunings
    meanNeuronResp(:, i) = alpha + beta * exp(gamma * (cos(pi / 90 * (i - phi))-1));
end

% open multi-core
if isempty(gcp) && nSimulations > 1
    pobj = parpool(20);
end

%% do it
[estVar_cTCNC, estVar_SFNC] = deal(zeros(length(nVxs_list), nCvxs, nSimulations));
for iSimu = 1:nSimulations
    iSimu
    stim = ceil((rand(1,nTrials)*180)); % we create a new set of stimuli in every simulation
    
    for iNvxs = 1:length(nVxs_list)  % loop different voxel size
        iNvxs
        W = 0.01 * 180 / nNeurons * rand(nVxs_list(iNvxs), nNeurons);
        % derive mean responses of voxels towards 180 stim
        meanVxsResp = W * meanNeuronResp;  % nVxs_list x nNeurons * nNeurons x stim = nVxs X nStim
        f = meanVxsResp; 
        % compute signal (tuning) correlation between voxels
        R_SC = corr(meanVxsResp'); %R_SC is a nVxs_list x nVxs_list correlation matrix
        
        %  cTCvxs
        R_cTCNC = R_SC;
        
        %   SFNC
        randOrder = Shuffle(1:nVxs_list(iNvxs));
        [xx,yy] = meshgrid(randOrder);
        ind = sub2ind(size(R_SC), xx, yy);
        R_SFNC = R_SC(ind);
        
        % set the vaiance matrix
        tau = gamrnd(vxsVarMean^2/vxsVarStd, vxsVarStd/vxsVarMean,[1,nVxs_list(iNvxs)]);
        varMat = sqrt(tau')*sqrt(tau);
        
        for iCvxs = 1:nCvxs % loop Cvxs
            % get covariance matrix
            % cTCvxs
            tmp = Cvxs(iCvxs)*R_cTCNC;
            tmp(logical(eye(size(R_cTCNC)))) = 1;
            Q_cTCNC = varMat.*tmp;
            
            % SFNC
            tmp = Cvxs(iCvxs)*R_SFNC;
            tmp(logical(eye(size(R_SFNC)))) = 1;
            Q_SFNC = varMat.*tmp;
            
            % now do the maximum likelihood estimator
            logpdf_cTCvxs = zeros(nTrials,180);
            logpdf_SFNC = zeros(nTrials,180);
            parfor j = 1:nTrials
                data_cTCvxs = mvnrnd(f(:,stim(j)), Q_cTCNC);
                data_SFNC = mvnrnd(f(:,stim(j)), Q_SFNC);
                for i=1:180  % do grid search
                    %logpdf(:,i) = log(mvnpdf(vpa(sym(data)),meanVxsResp(:,i)',diag(sqrt(meanVxsResp(:,i)))*tmp*diag(sqrt(meanVxsResp(:,i)))));
                    u = f(:,i);
                    logpdf_cTCvxs(j,i) = exp(-(data_cTCvxs'-u)' / Q_cTCNC * (data_cTCvxs'-u));
                    logpdf_SFNC(j,i) = exp(-(data_SFNC'-u)' / Q_SFNC * (data_SFNC'-u));
                end
            end
            assert(all(isfinite(logpdf_cTCvxs(:))));
            [~,y_pred] = max(logpdf_cTCvxs,[],2);% take the maximual likelihood
            estVar_cTCNC(iNvxs, iCvxs, iSimu) = circularsqerror(y_pred, stim', 180)/nTrials;
            
            assert(all(isfinite(logpdf_SFNC(:))));
            % take the maximual likelihood
            [~,y_pred] = max(logpdf_SFNC,[],2);
            estVar_SFNC(iNvxs,iCvxs, iSimu) = circularsqerror(y_pred, stim', 180)/nTrials;
        end
    end
end

%% close multicore
if ~isempty(gcp) 
    delete(gcp);
end
%% preprocess the data
estVar_SFNC_all =  estVar_SFNC;
estVar_cTCNC_all =  estVar_cTCNC;
estVar_SFNC = mean(estVar_SFNC,3);
estVar_cTCNC = mean(estVar_cTCNC,3);

%% plot results
% Make the legend labels
legend_labels = cell(1,nCvxs);
for i=1:nCvxs;legend_labels{i} = sprintf('c=%.02f',Cvxs(i)); end
legend_labels_nVxs = cell(1,length(nVxs_list));
for i=1:length(nVxs_list); legend_labels_nVxs{i}=sprintf('nVxs=%d',nVxs_list(i));end

close all;
h1 = cpsfigure(1,2);
set(h1,'Position',[0 0 800 300]);
ax(1) = subplot(1,2,1);
[lh,eh1] = myplot(nVxs_list,1./estVar_cTCNC',[], '-'); hold on;
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Number of voxels'); ylabel('Estimation efficiency');
title('TCvxs');
legend(legend_labels);
set(gca, 'XScale','log', 'YScale','log');

ax(2) = subplot(1,2,2);
[lh,eh2] = myplot(nVxs_list,1./estVar_SFNC',[], '-'); hold on;
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
title('SFNC');
xlabel('Number of voxels'); ylabel('Estimation efficiency');
set(gca, 'XScale','log', 'YScale','log');

% MSE/1 as a function of NC Coef
h2 = cpsfigure(1,2);
set(h2,'Position',[0 0 800 300]);
ax(1) = subplot(1,2,1);
[lh,eh1] = myplot(Cvxs,1./estVar_cTCNC,[], '-'); hold on;
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Cvxs'); ylabel('Estimation efficiency');
title('cTCvxs');
legend(legend_labels_nVxs);
set(gca,'YScale','log');

ax(2) = subplot(1,2,2);
[lh,eh2] = myplot(Cvxs,1./estVar_SFNC,[], '-'); hold on;
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
title('SFNC');
xlabel('Cvxs'); ylabel('Estimation efficiency');
set(gca,'YScale','log');

%% save data
if wantsave
    saveas(h1,'vxssimu_estimationMLE1.fig');
    saveas(h2,'vxssimu_estimationMLE2.fig');
    print(h1,'-dpdf','-painters','-r300','vxssimu_estimationMLE1.pdf');
    print(h2,'-dpdf','-painters','-r300','vxssimu_estimationMLE2.pdf');
    close all; save('vxssimu_estimationMLE.mat');
end