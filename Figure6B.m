%% In this script, we manipulate both neural and voxel-wise noise correlations on a stimulus-estimation task
% This script corresponds to Figure 6 panel B
% Running this script may take several minutes

clear all;close all;clc;

%%
nVxs = 200; % number of voxels
nNeurons = 180; % number of neurons
Cnc_vxs = [0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8, 0.99];  % NC coefficients for voxels
Cnc_neuron = [0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8, 0.99];  % NC coefficients for neurons
nSimulations = 10;
wantsave = 0;

nTrials = 1000; % how many trials per stim to simulate

% set the noise levels  
vxsVarMean = 3;
vxsVarStd = 1;

% Compute channel responses of 8 channels for 180 stim
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
% calculate neural 
R_SC_neuron = corr(meanNeuronResp');   % signal correlation matrix

% create variance matrix for all voxels
tau = gamrnd(vxsVarMean^2/vxsVarStd,vxsVarStd/vxsVarMean,[1, nVxs]);
varMat = sqrt(tau')*sqrt(tau);
%% open parallel
if isempty(gcp) && nSimulations > 1
    pobj  = parpool(20);
end
%%
estvar_cTCNC = cell(1, nSimulations);
for iSimu = 1:nSimulations
%parfor iSimu = 1:nSimulations
    iSimu
    stim = ceil((rand(1,nTrials)*180));
    estvar_cTCNCtmp = zeros(length(Cnc_neuron),length(Cnc_vxs));

    for iCnc_neuron = 1:length(Cnc_neuron)  % loop different voxel size
        iCnc_neuron
        
        W = 0.01 * 180/nNeurons * rand(nVxs, nNeurons); % Fix the W in one simulation
        meanVxsResp = W * meanNeuronResp;  % nVxs x nNeurons * nNeurons x stim = nVxs X nStim
        R_SC_vxs = corr(meanVxsResp'); %signalCorr is a nVxs x nVxs correlation matrix
        
        R_TCNC_neuron = Cnc_neuron(iCnc_neuron) * R_SC_neuron;
        R_TCNC_neuron(logical(eye(size(R_TCNC_neuron,1)))) = 1; % set diagnal to 1
        
        for iCnc_vxs = 1:length(Cnc_vxs) % loop Cnc_vxs
            % genereate voxel covariance matrix
            R_TCNC_vxs = Cnc_vxs(iCnc_vxs)*R_SC_vxs;
            R_TCNC_vxs(logical(eye(size(R_TCNC_vxs)))) = 1;
            Q_TCNC_vxs = varMat.*R_TCNC_vxs;
            
            % now do the maximum likelihood estimator
            logpdf = zeros(nTrials,180);
            parfor j = 1:nTrials % loop trials
                % generate neuronal responses in this trial
                meanNeuronResptmp = meanNeuronResp(:, stim(j));
                Q_TCNC_neuron = diag(sqrt(meanNeuronResptmp))* R_TCNC_neuron * diag(sqrt(meanNeuronResptmp));
                neurondata = posrect(mvnrnd(meanNeuronResptmp, Q_TCNC_neuron));                
                % generate voxel population response in this trial
                vxsdata = mvnrnd(W * neurondata', Q_TCNC_vxs);
                for i=1:180  % do grid search to get the loglikelihood
                    %logpdf(:,i) = log(mvnpdf(vpa(sym(vxsdata)),meanVxsResp(:,i)',diag(sqrt(meanVxsResp(:,i)))*tmp*diag(sqrt(meanVxsResp(:,i)))));
                    u = meanVxsResp(:,i); % expected voxel responses
                    logpdf(j,i) = -(vxsdata'-u)' / Q_TCNC_vxs * (vxsdata'-u); % mahalanobis distance
                end
            end
            assert(all(isfinite(logpdf(:))));
            % take the maximual likelihood
            [~,y_pred] = max(logpdf,[],2);
            estvar_cTCNCtmp(iCnc_neuron, iCnc_vxs, iSimu) = circularsqerror(y_pred, stim', 180)/nTrials;
        end        
    end
    estvar_cTCNC{iSimu} = estvar_cTCNCtmp;
end

if ~isempty(gcp)
    delete(gcp);
end
%%
% preprocess the vxsdata
estvar_cTCNC = cat(3,estvar_cTCNC{:});
mn_estvar_cTCNC = mean(estvar_cTCNC,3);
se_estvar_cTCNC = cat(3, zeros(size(mn_estvar_cTCNC)), (prctile(estvar_cTCNC,97.5,3) - prctile(estvar_cTCNC, 2.5, 3))/2);

%% plots
% make legend label
legend_label_Cncneuron = cell(1,length(Cnc_neuron));
for i=1:length(Cnc_neuron); legend_label_Cncneuron{i} = sprintf('Cnc\\_neuron=%.02f',Cnc_neuron(i)); end

% do the ploting
close all;
h1 = cpsfigure(1,1);
set(h1,'Position',[0 0 400 300]);

[lh,eh] = myplot(Cnc_vxs, 1./mn_estvar_cTCNC,[], '-'); hold on; 
%c = copper(length(lh));
c = parula(length(lh));
for i=1:length(lh)
    set(lh(i),'Color',c(i,:));
    if iscell(eh)
        set(eh{i},'Color',c(i,:));
    end
end
xlabel('Cnc\_vxs'); ylabel('Estimation efficiency');
title('cTCNC');
ylim([0.003, 0.04]);
set(gca,'YScale','log');
set(gca, 'YTick',[0.002,0.005,0.01,0.015,0.02])
legend(legend_label_Cncneuron);

%% save the result and the figure
if wantsave
    saveas(h1,'vxssimu_estimationMLE_neuralvxsNC1.fig');
    print(h1,'-dpdf','-painters','-r300','vxssimu_estimationMLE_neuralvxsNC1.pdf');
    close all;save('vxssimu_estimationMLE_neuralvxsNC.mat');
end
