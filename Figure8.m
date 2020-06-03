%% This code only take a few seconds 
clear all;close all;clc;

nVxs = 180;
nNeurons = 180;

anc = [0, 0.03, 0.1, 0.3, 0.5, 0.8, 0.99];

homoCoef = fliplr(1-10.^(linspace(-3,0,8)));
homoCoef = [homoCoef(2:end) 1];

wantsave = 1;

nSimulations = 10;

nAnc = length(anc);
legend_labels = cell(1,nAnc);

legend_label_nhomo = cell(1,length(homoCoef));
for i=1:length(homoCoef);legend_label_nhomo {i} = sprintf('Chomo%.03f',homoCoef(i)); end

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
%[independentSC_I, SC_I_ratio,shuffuleSC_I_ratio] = deal(zeros(nAnc, length(nVxs), nSimulations, length(oriStim))); % output structure cell array
[independentSC_I, SC_I_ratio,shuffuleSC_I_ratio] = deal(zeros(nAnc, length(homoCoef), nSimulations)); % output structure cell array
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
            tmp = rand(1, nNeurons-1);
            tmp = (1-homoCoef(ihomo)) * tmp;
            
            tmp2 = zeros(1,nNeurons);
            tmp2(i) = homoCoef(ihomo);
            tmp2(tmp2==0) = tmp;
            
            W(i, :)=tmp2;
        end
        
        %W = weightScale * W;
        
        
        % derive mean responses of voxels towards 180 stim
        [orienxx, phiyy] = meshgrid(oriStim, phi);
        f_derive = -pi/90*gamma*beta*exp(gamma*(cos((orienxx - phiyy)*pi/90) - 1)) .* ...
            sin(pi/90*(orienxx-phiyy));
        
        % calculate tuning curves
        meanVxsResp = W * meanNeuronResp;  % nVxs x nNeurons * nNeurons x stim = nVxs X nStim
        % normalize the tuning curve
        meanVxsResp = (meanVxsResp-min(meanVxsResp, [], 2))./(max(meanVxsResp,[],2)-min(meanVxsResp,[], 2)) * 19 + 1; % normalize response range between [1,20] to match neuronal response
                
        % set the vaiance matrix
        %tau = gamrnd(n_mean^2/n_std,n_std/n_mean,[1,nVxs]);
        tau = meanVxsResp(:, stim);
        varMat = sqrt(tau) * sqrt(tau');
        
        meanVxsResp_all{ihomo}=meanVxsResp;
        
        %
        deltaMeanVxsResp = W * f_derive;
        
        % compute signal (tuning) correlation between voxels
        signalCorr = corr(meanVxsResp'); %signalCorr is a nVxs x nVxs correlation matrix
        
        % calculate linear fisher information
        for iNcCoef = 1:length(anc) % loop across anc values
            % tuning-compatible correlatoin
            R_cTCNC = anc(iNcCoef)*signalCorr;
            R_cTCNC(logical(eye(size(R_cTCNC,1)))) = 1; % set diagnal to 1
            Q_cTCNC = R_cTCNC.*varMat;
            
            SC_I(iNcCoef,ihomo,iSimu) = deltaMeanVxsResp(:,stim)' / Q_cTCNC* deltaMeanVxsResp(:,stim) ;
        end
    end
end


%% average across simulation
SC_I_all = SC_I;
SC_I = mean(SC_I,3);
SC_I_norm = SC_I./repmat(SC_I(1,:), length(anc), 1);

%%

%%
close all;

h1=cpsfigure(1,3);
set(h1,'Position',[0 0 1200 300]);

ax(1)=subplot(1,3,1);
[lh,~]=myplot(anc,SC_I',[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Cnc'); ylabel('Information (deg-2)');
set(gca,'YScale','log', 'YLim', [1e-2, 1e2]);
title('cTCNC');
xlim([0,1]);
legend(legend_label_nhomo);

ax(2)=subplot(1,3,2);
[lh,~]=myplot(anc,SC_I_norm',[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Cnc'); ylabel('Normalized information (deg-2)');
set(gca,'YScale','log');
title('cTCNC');
xlim([0,1]);
legend(legend_label_nhomo);

ax(3)=subplot(1,3,3);
[lh,~]=myplot(1:180, meanVxsResp_all{1}(102,:),[], '-', 'Color',c(1,:));
[lh,~]=myplot(1:180, meanVxsResp_all{6}(125,:),[], '-', 'Color',c(6,:));
[lh,~]=myplot(1:180, meanVxsResp_all{end}(90,:),[], '-', 'Color',c(end,:));
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

