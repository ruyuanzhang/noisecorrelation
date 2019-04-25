%% simulate a pool of voxels on a stimulus-estimation task, and manipulate the heterogeneity of voxel tuning curves
clear all;close all;clc;

nVxs = 500;
nNeurons = 180;
Cvxs = [0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 0.9];
homoCoef = [0.03, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1];

%
n_mean = 3 * 40;  %% Note here, we scale the noise
n_std = 1; % note this parameter

wantsave = 1;
nSimulations = 10;

nCvxs = length(Cvxs);

% compute channel responses of n channels for 180 stim
phi = 0: 180 / nNeurons:180; % eight equally spaced orientation channels from 0-180.
phi = phi(2:end);
oriStim = 1:180;
channelResp = zeros(nNeurons, length(oriStim)); % 

% params tuning curves
alpha = 1;
beta = 19;
gamma = 2;
for i=1:180
    % use the same tuning curves as the real tunings
    meanNeuronResp(:, i) = alpha + beta * exp(gamma * (cos(pi / 90 * (i - phi))-1));
end

%%
%[independentSC_I, SC_I_ratio,shuffuleSC_I_ratio] = deal(zeros(nCvxs, length(nVxs), nSimulations, length(oriStim))); % output structure cell array
[independentSC_I, SC_I_ratio,shuffuleSC_I_ratio] = deal(zeros(nCvxs, length(homoCoef), nSimulations)); % output structure cell array
stim = 90;

% save the tuning curve
meanVxsResp_all = cell(1,length(homoCoef));
%meanresponse_all = 
for iSimu = 1:nSimulations % loop different simulation
    iSimu
    for ihomo=1:length(homoCoef)

        % derive the W
        W = zeros(nVxs, nNeurons);
        for i=1:nVxs
            idxtmp = randi(nNeurons);
            tmp = rand(1, nNeurons-1);
            tmp = (1-homoCoef(ihomo)) * tmp;
            
            tmp2 = zeros(1,nNeurons);
            tmp2(idxtmp) = homoCoef(ihomo);
            tmp2(tmp2==0) = tmp;
            
            W(i, :)=tmp2;
        end
        
        %W = weightScale * W;
        % set the vaiance matrix
        tau = gamrnd(n_mean^2/n_std,n_std/n_mean,[1,nVxs]);
        varMat = sqrt(tau')*sqrt(tau);
        
        % derive mean responses of voxels towards 180 stim
        [orienxx, phiyy] = meshgrid(oriStim, phi);
        f_derive = -pi/90*gamma*beta*exp(gamma*(cos((orienxx - phiyy)*pi/90) - 1)) .* ...
            sin(pi/90*(orienxx-phiyy));
        
        % calculate voxel tuning curves
        meanVxsResp = W * meanNeuronResp;  % nVxs x nNeurons * nNeurons x stim = nVxs X nStim
        % normalize
        meanVxsResp = (meanVxsResp-min(meanVxsResp, [], 2))./(max(meanVxsResp,[],2)-min(meanVxsResp,[], 2)) * 19 + 1; % normalize response range between [1,20] to match neuronal response
                
        meanVxsResp_all{ihomo}=meanVxsResp;
        %
        deltaMeanVxsResp = W * f_derive;
        
        % compute signal (tuning) correlation between voxels
        signalCorr = corr(meanVxsResp'); %signalCorr is a nVxs x nVxs correlation matrix
        
        % calculate linear fisher information
        for iNcCoef = 1:length(Cvxs) % loop across Cvxs values
            % tuning-compatible correlatoin
            R_cTCNC = Cvxs(iNcCoef)*signalCorr;
            R_cTCNC(logical(eye(size(R_cTCNC,1)))) = 1; % set diagnal to 1
            Q_cTCNC = R_cTCNC.*varMat;
            
            SC_I(iNcCoef,ihomo,iSimu) = deltaMeanVxsResp(:,stim)' / scCov * deltaMeanVxsResp(:,stim) ;
        end
    end
end


%% average across simulation
SC_I_all = SC_I;
SC_I = mean(SC_I,3);
SC_I_norm = SC_I./repmat(SC_I(1,:),length(homoCoef),1);

%%

%% plot results
% Make legend labels
legend_label_nhomo = cell(1,length(homoCoef));
for i=1:length(homoCoef);legend_label_nhomo {i} = sprintf('Chomo%.02f',homoCoef(i)); end

close all;
h1=cpsfigure(1,3);
set(h1,'Position',[0 0 1200 300]);

ax(1)=subplot(1,3,1);
[lh,~]=myplot(Cvxs,SC_I',[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('NC coef'); ylabel('Information');
set(gca,'YScale','log');
title('cTCNC');
xlim([0,1]);
legend(legend_label_nhomo);

ax(2)=subplot(1,3,2);
[lh,~]=myplot(Cvxs,SC_I_norm',[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('NC coef'); ylabel('Normalized information');
%set(gca,'YScale','log');
title('cTCNC');
xlim([0,1]);
legend(legend_label_nhomo);

ax(3)=subplot(1,3,3);
[lh, ~]=myplot(1:180, meanVxsResp_all{1}(102,:),[], '-', 'Color',c(1,:));
[lh, ~]=myplot(1:180, meanVxsResp_all{6}(125,:),[], '-', 'Color',c(6,:));
[lh, ~]=myplot(1:180, meanVxsResp_all{end}(200,:),[], '-', 'Color',c(end,:));
xlabel('Orientation'); ylabel('Response (a.u.)');
title('cTCNC');
xlim([0,180]);
legend(legend_label_nhomo{[1,6,8]});

%% Save the figure
if wantsave
    saveas(h1, 'vxssimu_estimation_calclif_titrate1.fig');
    print(h1,'-dpdf','-painters','-r300', 'vxssimu_estimation_calclif_titrate1.pdf');
    close all;save('vxssimu_estimation_calclif_titrate.mat');
end

