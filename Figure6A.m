%% In this script, we manipulate both neural and voxel-wise noise correlations on a stimulus-classification task
% This script corresponds to Figure 6 panel A
% Running this script takes around 5 mins

%%
clear all;close all;clc;
nVxs = 200; % number of voxels in the model
nNeurons = 180; % number of neurons in the model 
Cnc_vxs = [0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8, 0.99];  % NC coefficients for voxels
Cnc_neuron = [0, 0.03, 0.1, 0.3, 0.5, 0.8, 0.99];  % NC coefficients for neurons
nSimulations = 10;
stim1 = 80;  
stim2 = 100;
nTrials = 2000; % how many trials per stim to simulate
wantsave = 1;

% Noise levels at the voxel level
vxsVarMean = 3;
vxsVarStd = 1;

% compute channel responses of 8 channels for 180 stim
phi = 0: 180 / nNeurons:180; % eight equally spaced orientation channels from 0-180.
phi = phi(2:end);
oriStim = 1:180;
channelResp = zeros(nNeurons, length(oriStim)); % 

% Params tuning curves
alpha = 1;
beta = 19;
gamma = 2;
meanNeuronResp = zeros(nNeurons,180);
for i=1:180
    % use the same tuning curves as the real tunings
    meanNeuronResp(:, i) = alpha + beta * exp(gamma * (cos(pi / 90 * (i - phi))-1));
end
% calculate neural 
R_SC_neuron = corr(meanNeuronResp');   % signal correlation matrix
% generate neuron responses
populationMean1 = meanNeuronResp(:, stim1);
populationMean2 = meanNeuronResp(:, stim2);

% create variance matrix for all voxels
tau = gamrnd(vxsVarMean^2/vxsVarStd, vxsVarStd/vxsVarMean,[1, nVxs]);
varMat = sqrt(tau')*sqrt(tau);
%% open parallel
% if isempty(gcp) && nSimulations>1
%     pobj  = parpool;
% end
%%
pCorrect_cTCNC = cell(1, nSimulations);
for iSimu = 1:nSimulations
    iSimu
    pCorrect_cTCNCtmp = zeros(length(Cnc_neuron),length(Cnc_vxs));
    
    for iCnc_neuron = 1:length(Cnc_neuron)  % loop different voxel size
        
        W = 0.01 * 180/nNeurons * rand(nVxs, nNeurons); % Fix the W in one simulation
        meanVxsResp = W * meanNeuronResp;  % nVxs x nNeurons * nNeurons x stim = nVxs X nStim
        R_SC_vxs = corr(meanVxsResp'); %signalCorr is a nVxs x nVxs correlation matrix
        
        R_TCNC_neuron = Cnc_neuron(iCnc_neuron) * R_SC_neuron;
        R_TCNC_neuron(logical(eye(size(R_TCNC_neuron,1)))) = 1; % set diagnal to 1
        
        %% tuning compatible noise correlation
        for iCnc_vxs = 1:length(Cnc_vxs) % loop Cnc_vxs
            iCnc_vxs
            % generate neuron responses
            
            Q_TCNC_neuron1 = diag(sqrt(populationMean1))* R_TCNC_neuron * diag(sqrt(populationMean1));
            Q_TCNC_neuron2 = diag(sqrt(populationMean2))* R_TCNC_neuron * diag(sqrt(populationMean2));
            
            neurondata1 = posrect(mvnrnd(populationMean1, Q_TCNC_neuron1, nTrials));
            neurondata2 = posrect(mvnrnd(populationMean2, Q_TCNC_neuron2, nTrials));

            vxsdataTrialMean1 = neurondata1 * (W'); % linear transform the neuron data to vxs resp
            vxsdataTrialMean2 = neurondata2 * (W');
            
            %clear neurondata1 neurondata2;
            
            % generate voxel data and classify
            R_TCNC_vxs = Cnc_vxs(iCnc_vxs)*R_SC_vxs;
            R_TCNC_vxs(logical(eye(size(R_TCNC_vxs)))) = 1;
            Q_TCNC_vxs = varMat.*R_TCNC_vxs;
            
            clear R_TCNC_vxs
            
            vxsdata1 = zeros(nTrials, nVxs);
            vxsdata2 = zeros(nTrials, nVxs);
            parfor iTrial = 1:nTrials
                vxsdata1(iTrial, :) = mvnrnd(vxsdataTrialMean1(iTrial, :), Q_TCNC_vxs); % vxs resp for stim1
                vxsdata2(iTrial, :) = mvnrnd(vxsdataTrialMean2(iTrial, :), Q_TCNC_vxs); % vxs resp for stim2
            end
            
%            vxsdata1 = mvnrnd(meanVxsResp(:, stim1), Q_TCNC_vxs, nTrials); % vxs resp for stim1
%            vxsdata2 = mvnrnd(meanVxsResp(:, stim2), Q_TCNC_vxs, nTrials); % vxs resp for stim2

%            vxsdata1 = neurondata1; % linear transform the neuron data to vxs resp
%            vxsdata2 = neurondata2; % linear transform the neuron data to vxs resp

            sample = [vxsdata1(1:nTrials/2,:); vxsdata2(1:nTrials/2,:)];
            training = [vxsdata1(nTrials/2+1:end,:); vxsdata2(nTrials/2+1:end,:)];
            group = [1*ones(nTrials/2,1); 2*ones(nTrials/2,1)];
            
            [class,~,~,~,classObj] = classify(sample,training,group,'linear');
            
            good = 0;
            good = good + sum(class(1:nTrials/2)==1);
            good = good + sum(class(nTrials/2+1:end)==2);
            pCorrect_cTCNCtmp(iCnc_neuron, iCnc_vxs) = good/nTrials * 100;
        end
        
    end
    pCorrect_cTCNC{iSimu} = pCorrect_cTCNCtmp;
end
%%
% preprocess the data
pCorrect_cTCNC = cat(3,pCorrect_cTCNC{:});
mn_pCorrect_cTCNC = mean(pCorrect_cTCNC,3);
se_pCorrect_cTCNC = cat(3, zeros(size(mn_pCorrect_cTCNC)), (prctile(pCorrect_cTCNC,97.5,3)-prctile(pCorrect_cTCNC,2.5,3))/2);

%% plot the results
%
legend_label_Cncneuron = cell(1,length(Cnc_neuron));
for i=1:length(Cnc_neuron); legend_label_Cncneuron{i} = sprintf('Cnc\\_neuron=%.02f',Cnc_neuron(i)); end

close all;
h1 = cpsfigure(1,1);
set(h1,'Position',[0 0 400 300]);

ax(1) = subplot(1,1,1);
[lh,eh] = myplot(Cnc_vxs,mn_pCorrect_cTCNC,[], '-'); hold on; 
%c = copper(length(lh));
c = parula(length(lh));
for i=1:length(lh)
    set(lh(i),'Color',c(i,:));
    if iscell(eh)
        set(eh{i},'Color',c(i,:));
    end
end
xlabel('Cnc\_vxs'); ylabel('Classification accuracy (%)');
title('cTCNC');
legend(legend_label_Cncneuron);
ylim([50 100]);

%% save the result and the figure
if wantsave
    saveas(h1,'vxssimu_classify_neuralvxsNC1.fig');
    print(h1,'-dpdf','-painters','-r300','vxssimu_classify_neuralvxsNC1.pdf');
    close all;save('vxssimu_classify_neuralvxsNC.mat');
end
