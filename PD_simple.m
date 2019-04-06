% PD_simple.m
%
% Description: Utilizing parameters extracted from voice recordings of
% several patients with and without Parkinson's Disease (PD), this simple
% neural network classifies YES PD or NO PD based on 2 of the parameters,
% Jitter % and Shimmer %.
%
% Authors: Braeden Benedict & Ben Wdowik
% Course: EE47058 (Neural Networks), University of Notre Dame
% Date: February 2019

%% Clear
clear all
clc

%% Initialize Variables
n_patients_nPD = 8; %no PD
n_patients_yPD = 23; %yes PD
n_patients_tot = n_patients_nPD + n_patients_yPD; %total patients

%% Import Data
%import
data_file = 'parkinsons_data.csv';
PD_data_unsorted = readtable(data_file);

%reorder data by status
PD_data_sorted = sortrows(PD_data_unsorted,'status');

%break into 2x inputs and 1x output
PD_data_in1 = table2array(PD_data_sorted(:,5)); %input 1 (5=jitter)
PD_data_in2 = table2array(PD_data_sorted(:,10)); %input 2 (10=shimmer)
PD_data_out = table2array(PD_data_sorted(:,18)); %correct output (18=status)

%% Create Training and Testing Sets
%split patients in half
range_nPD = [1 (find(PD_data_out,1) - 1)];
range_yPD = [find(PD_data_out,1) length(PD_data_out)];

%data range for training
train_range_nPD = [range_nPD(1) ceil(range_nPD(2)/2)];
train_range_yPD = [range_yPD(1) (ceil((range_yPD(2)-range_yPD(1))/2))+range_yPD(1)];

%data range for training
test_range_nPD = [train_range_nPD(2)+1 range_nPD(2)];
test_range_yPD = [train_range_yPD(2)+1 range_yPD(2)];

%isolate TRAINING sets based on ranges
train_data_in1 = [PD_data_in1(train_range_nPD(1):train_range_nPD(2));...
                        PD_data_in1(train_range_yPD(1):train_range_yPD(2))];
train_data_in2 = [PD_data_in2(train_range_nPD(1):train_range_nPD(2));...
                        PD_data_in2(train_range_yPD(1):train_range_yPD(2))];
train_data_out = [PD_data_out(train_range_nPD(1):train_range_nPD(2));...
                        PD_data_out(train_range_yPD(1):train_range_yPD(2))];
                    
%isolate TESTING sets based on ranges
test_data_in1 = [PD_data_in1(test_range_nPD(1):test_range_nPD(2));...
                        PD_data_in1(test_range_yPD(1):test_range_yPD(2))];
test_data_in2 = [PD_data_in2(test_range_nPD(1):test_range_nPD(2));...
                        PD_data_in2(test_range_yPD(1):test_range_yPD(2))];
test_data_out = [PD_data_out(test_range_nPD(1):test_range_nPD(2));...
                        PD_data_out(test_range_yPD(1):test_range_yPD(2))];
                    
%% Create and Train Neural Network

figure(1)
clf(1)
hold on
new_train_data_out = [];
new_train_data_in1 = [];
new_train_data_in2 = [];



for j=1:length(train_data_out)
    if train_data_out(j) == 0
        new_train_data_out = [new_train_data_out; train_data_out(j)];
        new_train_data_in1 = [new_train_data_in1; train_data_in1(j)];
        new_train_data_in2 = [new_train_data_in2; train_data_in2(j)];
        plot(train_data_in1(j),train_data_in2(j),'.b','MarkerSize',10); %healthy
    elseif (train_data_out(j) == 1)
        new_train_data_out = [new_train_data_out; train_data_out(j)];
        new_train_data_in1 = [new_train_data_in1; train_data_in1(j)];
        new_train_data_in2 = [new_train_data_in2; train_data_in2(j)];
        plot(train_data_in1(j),train_data_in2(j),'.r','MarkerSize',10); %PD
    end
end

training_inputs = [new_train_data_in1 new_train_data_in2]';
training_outputs = new_train_data_out';
save('basic_parkinsons', 'training_inputs','training_outputs')
tic
net1 = perceptron('hardlim','learnp');
net1 = train(net1,training_inputs,training_outputs);
net1.b{1}

%% Plot Training Data and Network
b = net1.b{1}; %bias
W = net1.IW{1}; %weights

nn_line_x = linspace(0,0.014);
nn_line_y = (-W(1)*nn_line_x-b)/W(2);


plot(nn_line_x,nn_line_y,'m') %plot linear boundary


grid on
xlabel('Jitter')
ylabel('Shimmer')
title('Training Data')
toc




%% Plot all data
% figure(2)
% clf(2)
% 
% hold on
% 
% for j=1:length(PD_data_status)
%     if PD_data_status(j) == 0
%         plot(PD_data_jitter(j),PD_data_shimmer(j),'.b','MarkerSize',10); %healthy
%     else
%         plot(PD_data_jitter(j),PD_data_shimmer(j),'.r','MarkerSize',10); %PD
%     end
% end
% 
% grid on
% xlabel('Jitter')
% ylabel('Shimmer')
% title('All Data')