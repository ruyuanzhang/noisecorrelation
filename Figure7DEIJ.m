%% Simulate a pool of voxels and calculate information in a stimulus-estimation task
% This scripts correspond to Figure 7 panels DEIJ
% Runing this scripts takes a few minutes

%%
clear all;close all;clc;

nVxs_list = [10, 20, 50, 100, 200, 500];
Cvxs = [0,0.01,0.03,0.1,0.3,0.5,0.7,0.9];  
nNeurons = 180;
nCnc = length(Cvxs); 
nSimulations = 100;
wantsave = 1;

vxsVarMean = 3;
vxsVarStd = 1;
%% compute channel responses of 8 channels for 180 stim
phi = 0: 180 / nNeurons:180; % eight equally spaced orientation channels from 0-180.
phi = phi(2:end);
oriStim = 1:180;
[orienxx, phiyy] = meshgrid(oriStim, phi);
meanNeuronResp = zeros(nNeurons, length(oriStim)); % 
% params tuning curves
alpha = 1;
beta = 19;
gamma = 2;
for i=1:180
    % use the same tuning curves as the real tunings
    meanNeuronResp(:, i) = alpha + beta * exp(gamma * (cos(pi / 90 * (i - phi))-1));
end
f_derive = -pi/90*gamma*beta*exp(gamma*(cos((orienxx - phiyy)*pi/90) - 1)) .* ...
    sin(pi/90*(orienxx - phiyy));
%
legend_labels = cell(1,nCnc);
for i=1:nCnc;legend_labels{i} = sprintf('c=%.02f',Cvxs(i)); end
legend_labels_nVxs = cell(1,length(nVxs_list));
for i=1:length(nVxs_list); legend_labels_nVxs{i}=sprintf('nVxs=%d',nVxs_list(i));end

%% open multi-core
if isempty(gcp) && nSimulations > 1
    pobj = parpool(20);
end

%% do it
[Info_cTCNC, Info_SFNC] = deal(cell(1,nSimulations)); % for parallel computing purposes
parfor iSimu = 1:nSimulations
    iSimu
    [Info_cTCNCtmp, Info_SFNCtmp] = deal(zeros(length(nVxs_list), nCnc));

    for iNvxs = 1:length(nVxs_list)  % loop different voxel size
        W = 0.01 * 180/nNeurons * rand(nVxs_list(iNvxs), nNeurons);
        % derive mean responses of voxels towards 180 stim
        meanVxsResp = W * meanNeuronResp;  % nVxs_list x nNeurons * nNeurons x stim = nVxs X nStim
        meanVxsResp_derive = W * f_derive;
        
        % compute signal (tuning) correlation between voxels
        R_SC = corr(meanVxsResp'); %R_SC is a nVxs_list x nVxs_list correlation matrix
        
        %  generate cTCNC noise correlation matrix
        R_cTCNC = R_SC;
        
        % generate SFNC noise correlation matrix
        randOrder = Shuffle(1:nVxs_list(iNvxs));
        [xx,yy] = meshgrid(randOrder);
        ind = sub2ind(size(R_SC), xx, yy);
        R_SFNC = R_SC(ind);
        
        % set the vaiance matrix
        tau = gamrnd(vxsVarMean^2/vxsVarStd, vxsVarStd/vxsVarMean,[1,nVxs_list(iNvxs)]);
        varMat = sqrt(tau')*sqrt(tau);
        
        for iCnc = 1:nCnc % loop Cvxs
            % get covariance matrix
            % cTCNC covariance
            tmp = Cvxs(iCnc)*R_cTCNC;
            tmp(logical(eye(size(tmp)))) = 1;
            Q_cTCNC = varMat.*tmp;
            
            % SFNC covariance
            tmp = Cvxs(iCnc)*R_SFNC;
            tmp(logical(eye(size(tmp)))) = 1;
            Q_SFNC = varMat.*tmp;
            
            % now do the maximum likelihood estimator
            Info_cTCNCtmp1 = 0;
            Info_SFNCtmp1 = 0;
            for i = 1:180 % loop 180 orientations                
                Info_cTCNCtmp1 =  Info_cTCNCtmp1 + meanVxsResp_derive(:, i)' / Q_cTCNC * meanVxsResp_derive(:, i);
                Info_SFNCtmp1 =  Info_SFNCtmp1 + meanVxsResp_derive(:, i)' / Q_SFNC * meanVxsResp_derive(:, i);
            end
            Info_cTCNCtmp(iNvxs,iCnc) = Info_cTCNCtmp1 / 180; % information per simulus
            Info_SFNCtmp(iNvxs,iCnc) = Info_SFNCtmp1 / 180;            
        end
    end
    Info_cTCNC{iSimu} = Info_cTCNCtmp; % for parallel computing purpose;
    Info_SFNC{iSimu} = Info_SFNCtmp; 
end

%% close multicore
if ~isempty(gcp) 
    delete(gcp);
end
%% preprocess the data
Info_SFNC_all =  cat(3, Info_SFNC{:});
Info_cTCNC_all =  cat(3, Info_cTCNC{:});
Info_SFNC = mean(Info_SFNC_all,3);
Info_cTCNC = mean(Info_cTCNC_all,3);

%%
close all;
h1 = cpsfigure(1,2);
set(h1,'Position',[0 0 800 300]);
ax(1) = subplot(1,2,1);
[lh,eh1] = myplot(nVxs_list, Info_cTCNC',[], '-'); hold on;
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Number of voxels'); ylabel('Information');
title('TCNC');
legend(legend_labels);
set(gca, 'XScale','log', 'YScale','log');

ax(2) = subplot(1,2,2);
[lh, eh2] = myplot(nVxs_list, Info_SFNC',[], '-'); hold on;
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
title('SFNC');
xlabel('Number of voxels'); ylabel('Information');
set(gca, 'XScale','log', 'YScale','log');

h2 = cpsfigure(1,2);
set(h2,'Position',[0 0 800 300]);
ax(1) = subplot(1,2,1);
[lh,eh1] = myplot(Cvxs, Info_cTCNC,[], '-'); hold on;
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Cvxs'); ylabel('Information');
title('cTCNC');
legend(legend_labels_nVxs);
set(gca, 'YScale','log');

ax(2) = subplot(1,2,2);
[lh,eh2] = myplot(Cvxs, Info_SFNC,[], '-'); hold on;
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
title('SFNC');
xlabel('Cvxs'); ylabel('Information');
set(gca, 'YScale','log');

%% save data
if wantsave
    saveas(h1,'vxssimu_estimation_calclfi1.fig');
    saveas(h2,'vxssimu_estimation_calclfi2.fig');
    print(h1,'-dpdf','-painters','-r300','vxssimu_estimation_calclfi1.pdf');
    print(h2,'-dpdf','-painters','-r300','vxssimu_estimation_calclfi2.pdf');
    close all; save('vxssimu_estimation_calclfi.mat');
end