%% Simulate a pool of orientation-selective neurons and calculate information in a stimulus-estimation task
% This scripts correspond to Figure 7 panels ABCFGH
% Runing this scripts takes a couple seconds

clear all;close all;clc;

nNeurons_list = [10, 20, 50, 100, 200, 400, 1000, 2000];
Cneuron = [0, 0.03, 0.1, 0.3, 0.5, 0.8, 0.99]; % noise correlation coefficient

wantsave = 0;

% params for the tuning curve
alpha = 1;
beta = 19;
gamma = 2;
orien = 1 :180;  % deg, possible orientation stimuli


[Info_aTCNC, Info_cTCNC, Info_SFNC] = deal(zeros(length(nNeurons_list),length(Cneuron)));

for iNeuron = 1:length(nNeurons_list)
    iNeuron
    
    nNeurons = nNeurons_list(iNeuron);
    % generate tuning curves
    phi = 0:180/nNeurons:180;  % deg, prefered orientation
    phi = phi(2:end);
    % von mises tuning curve
    [orienxx, phiyy] = meshgrid(orien, phi);
    meanNeuronResp = alpha + beta * exp(gamma*(cos((orienxx - phiyy)*pi/90) - 1)); % nNeurons x nOrientation responses
    
    meanNeuronResp_derive = -pi/90*gamma*beta*exp(gamma*(cos((orienxx - phiyy)*pi/90) - 1)) .* ...
    sin(pi/90*(orienxx-phiyy));
    
    for iCnc = 1:length(Cneuron)        
        % ==== angular-based TCNC ====
        L = 1; % control the 
        [phix, phiy] = meshgrid(phi, phi);
        orienDiff = abs(circulardiff(phix, phiy, 180));
        R_aTCNC = Cneuron(iCnc)*exp(-orienDiff*pi/180/L);
        R_aTCNC(logical(eye(size(R_aTCNC,1)))) = 1;
        
        % ==== curve-based TCNC====
        R_SC = corr(meanNeuronResp');   % signal correlation matrix
        R_cTCNC = Cneuron(iCnc) * R_SC;
        R_cTCNC(logical(eye(size(R_cTCNC, 1)))) = 1; % set diagnal to 1
        
        % ==== SFNC ====
        randOrder = Shuffle(1:nNeurons);
        [xx,yy] = meshgrid(randOrder);
        ind = sub2ind(size(R_SC), xx, yy);
        R_SFNC = R_SC(ind);
        R_SFNC = Cneuron(iCnc) * R_SFNC;
        R_SFNC(logical(eye(size(R_SFNC,1)))) = 1; % set diagnal to 1
        
        % ==== now calculate Fisher information ========
        Info_aTCNCtmp = 0;
        Info_cTCNCtmp = 0;
        Info_SFNCtmp = 0;
        for i = 1:180 % loop over all 180 orientations
            Q_aTCNC = diag(sqrt(meanNeuronResp(:,i))) * R_aTCNC * diag(sqrt(meanNeuronResp(:,i)));
            Info_aTCNCtmp = Info_aTCNCtmp +  meanNeuronResp_derive(:,i)'/ Q_aTCNC * meanNeuronResp_derive(:,i);
            
            Q_cTCNC = diag(sqrt(meanNeuronResp(:,i))) * R_cTCNC * diag(sqrt(meanNeuronResp(:,i)));
            Info_cTCNCtmp = Info_cTCNCtmp +  meanNeuronResp_derive(:,i)'/ Q_cTCNC * meanNeuronResp_derive(:,i);
            
            Q_SFNC = diag(sqrt(meanNeuronResp(:,i))) * R_SFNC * diag(sqrt(meanNeuronResp(:,i)));
            Info_SFNCtmp = Info_SFNCtmp +  meanNeuronResp_derive(:,i)'/ Q_SFNC * meanNeuronResp_derive(:,i);
        end
        Info_aTCNC(iNeuron, iCnc) = Info_aTCNCtmp/180;
        Info_cTCNC(iNeuron, iCnc) = Info_cTCNCtmp/180;
        Info_SFNC(iNeuron, iCnc) = Info_SFNCtmp/180;
    end
end
%% plot result
% make legend labels
legend_label = cell(1,length(Cneuron));
for i=1:length(Cneuron); legend_label{i}=sprintf('C0=%.2f',Cneuron(i));end
legend_label_nNeuron = cell(1,length(nNeurons_list));
for i=1:length(nNeurons_list); legend_label_nNeuron{i}=sprintf('nNeuron=%d',nNeurons_list(i));end

close all;
h1=cpsfigure(1,3); % 
set(h1,'Position',[0 0 1200 300]);
ax(1)=subplot(1,3,1);
[lh,~]=myplot(nNeurons_list,Info_aTCNC',[],'-');
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Number of neurons'); ylabel('Information');
set(gca, 'XScale','log','YScale','log');
title('aTCNC');
legend(legend_label);

ax(2)=subplot(1,3,2);
[lh,~]=myplot(nNeurons_list,Info_cTCNC',[],'-');
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Number of neurons'); ylabel('Information');
set(gca, 'XScale','log','YScale','log');
title('cTCNC');

ax(3)=subplot(1,3,3);
[lh,~]=myplot(nNeurons_list,Info_SFNC',[],'-');
c = cool(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Number of neurons'); ylabel('Information');
set(gca, 'XScale','log','YScale','log');
title('SFNC');

h2 = cpsfigure(1,3);
set(h2,'Position',[0 0 1200 300]);
ax(1)=subplot(1,3,1);
[lh,~]=myplot(Cneuron,Info_aTCNC,[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Cneuron'); ylabel('Information');
set(gca, 'YScale','log');
title('aTCNC');
legend(legend_label_nNeuron);

ax(2)=subplot(1,3,2);
[lh,~]=myplot(Cneuron,Info_cTCNC,[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Cneuron'); ylabel('Information');
set(gca,'YScale','log');
title('cTCNC');

ax(3)=subplot(1,3,3);
[lh,~]=myplot(Cneuron,Info_SFNC,[],'-');
c = parula(length(lh));
for i=1:length(lh);set(lh(i),'Color',c(i,:));end
xlabel('Cneuron'); ylabel('Information');
set(gca, 'YScale','log');
title('SFNC');

%%
if wantsave
    saveas(h1,'neuronsimu_estimation_calclfi1.fig');
    saveas(h2,'neuronsimu_estimation_calclfi2.fig');
    print(h1, '-dpdf', '-painters', '-r300', 'neuronsimu_estimation_calclfi1.pdf');
    print(h2, '-dpdf', '-painters', '-r300', 'neuronsimu_estimation_calclfi2.pdf');
    close all; save('neuronsimu_estimation_calclfi.mat');
end