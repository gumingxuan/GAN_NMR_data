clear; close all; clc;

% Read CSV file
data = readtable('FUEL_loss_all.csv');

% Calculate the average loss for each Epoch
numBatchesPerEpoch = 8;  % Assuming there are 8 Batches per Epoch
epochs = unique(data.Epoch);  % Get unique values for all Epochs

% Initialize arrays for average losses
avg_g1_loss = zeros(length(epochs), 1);
avg_g2_loss = zeros(length(epochs), 1);
avg_g3_loss = zeros(length(epochs), 1);
avg_d1_loss = zeros(length(epochs), 1);
avg_d2_loss = zeros(length(epochs), 1);
avg_d3_loss = zeros(length(epochs), 1);

% Calculate the average loss for each Epoch
for i = 1:length(epochs)
    epoch_data = data(data.Epoch == epochs(i), :);
    avg_g1_loss(i) = mean(epoch_data.G1Loss);
    avg_g2_loss(i) = mean(epoch_data.G2Loss);
    avg_g3_loss(i) = mean(epoch_data.G3Loss);
    avg_d1_loss(i) = mean(epoch_data.D1Loss);
    avg_d2_loss(i) = mean(epoch_data.D2Loss);
    avg_d3_loss(i) = mean(epoch_data.D3Loss);
end


figure;
plot(epochs+1, avg_g1_loss,'b', 'LineWidth', 3);hold on;
plot(epochs+1, avg_g2_loss,'g', 'LineWidth', 3);hold on;
plot(epochs+1, avg_g3_loss,'r', 'LineWidth', 3);hold on;
xlabel('Epoch','fontsize',20,'fontweight','bold','fontname','Times New Roman')
ylabel('Loss','fontsize',20,'fontweight','bold','fontname','Times New Roman')
set(gca,'fontsize',24,'fontweight','bold','fontname','Times New Roman')
h=legend('Generator 1','Generator 2','Generator 3','location','northwest');
set(h,'fontsize',15,'fontweight','bold','fontname','Times New Roman', 'Box', 'off')
xticks(min(epochs):1000:max(epochs+1));
xlim([0 5000])


figure;
plot(epochs+1, avg_d1_loss,'b', 'LineWidth', 3);hold on;
plot(epochs+1, avg_d2_loss,'g', 'LineWidth', 3);hold on;
plot(epochs+1, avg_d3_loss,'r', 'LineWidth', 3);hold on;
xlabel('Epoch','fontsize',20,'fontweight','bold','fontname','Times New Roman')
ylabel('Loss','fontsize',20,'fontweight','bold','fontname','Times New Roman')
set(gca,'fontsize',24,'fontweight','bold','fontname','Times New Roman')
h=legend('Discriminator 1','Discriminator 2','Discriminator 3','location','southeast');
set(h,'fontsize',15,'fontweight','bold','fontname','Times New Roman', 'Box', 'off')
legend show;
xticks(min(epochs):1000:max(epochs+1));
xlim([0 5000])







%%  
figure;
plot(epochs+1, avg_g1_loss,'b', 'LineWidth', 3);hold on;
plot(epochs+1, avg_g2_loss,'g', 'LineWidth', 3);hold on;
plot(epochs+1, avg_g3_loss,'r', 'LineWidth', 3);hold on;
xlabel('Epoch','fontsize',20,'fontweight','bold','fontname','Times New Roman')
ylabel('Loss','fontsize',20,'fontweight','bold','fontname','Times New Roman')
set(gca,'fontsize',24,'fontweight','bold','fontname','Times New Roman')
h=legend('Generator 1','Generator 2','Generator 3','location','northeast');
set(h,'fontsize',15,'fontweight','bold','fontname','Times New Roman', 'Box', 'off')

xticks(min(epochs):1000:max(epochs+1));
xlim([0 5000])
ylim([0 8])


figure;
plot(epochs+1, avg_d1_loss,'b', 'LineWidth', 3);hold on;
plot(epochs+1, avg_d2_loss,'g', 'LineWidth', 3);hold on;
plot(epochs+1, avg_d3_loss,'r', 'LineWidth', 3);hold on;
xlabel('Epoch','fontsize',20,'fontweight','bold','fontname','Times New Roman')
ylabel('Loss','fontsize',20,'fontweight','bold','fontname','Times New Roman')
set(gca,'fontsize',24,'fontweight','bold','fontname','Times New Roman')
h=legend('Discriminator 1','Discriminator 2','Discriminator 3','location','northeast');
set(h,'fontsize',15,'fontweight','bold','fontname','Times New Roman', 'Box', 'off')

xticks(min(epochs):1000:max(epochs+1));
xlim([0 5000])
ylim([0 3])










