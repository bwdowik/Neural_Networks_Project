% PD_hebbian.m
%
% Description: Determines a decision boundary from a PD
% dataset using jitter and shimmer values, using a Hebbian
% learning approach using the pseudoinverse rule
%
% Authors: Braeden Benedict & Ben Wdowik
% Course: EE47058 (Neural Networks), University of Notre Dame
% Date: March 2019

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
                    
%% Plot healthy and unhealthy training data
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
    %Ensures data is linearly separable for simplification
    elseif (train_data_out(j) == 1) && (train_data_in1(j)>0.006)
        new_train_data_out = [new_train_data_out; train_data_out(j)];
        new_train_data_in1 = [new_train_data_in1; train_data_in1(j)];
        new_train_data_in2 = [new_train_data_in2; train_data_in2(j)];
        plot(train_data_in1(j),train_data_in2(j),'.r','MarkerSize',10); %PD
    end
end

grid on
xlabel('Jitter')
ylabel('Shimmer')
title('Training Data')

training_inputs = [new_train_data_in1 new_train_data_in2]';
training_outputs = new_train_data_out';

%% Hebbian approach- training
T = training_outputs.*2 -1; %making -1, 1 instead of 0,1
P = training_inputs;
P(3,:) = ones(1,length(P(1,:))); %for a bias weight

P_pseudo = P'*inv(P*P');
W = T*P_pseudo;
bias = W(3);

Wp = W*P;
A = hardlims(Wp) %classification results

nn_line_x = linspace(0,0.014);
nn_line_y = (-W(1)*nn_line_x-bias)/W(2);
plot(nn_line_x,nn_line_y,'m') %plot linear boundary
hold off

%% Testing
% figure(2)
% clf(2)
% hold on
% new_test_data_out = [];
% new_test_data_in1 = [];
% new_test_data_in2 = [];
% for j=1:length(test_data_out)
%     if test_data_out(j) == 0
%         plot(test_data_in1(j),test_data_in2(j),'.b','MarkerSize',10); %healthy
%     elseif (test_data_out(j) == 1)
%         plot(test_data_in1(j),test_data_in2(j),'.r','MarkerSize',10); %PD
%     end
% end
% 
% grid on
% xlabel('Jitter')
% ylabel('Shimmer')
% title('Testing Data')
% 
% P = [test_data_in1 test_data_in2]';
% P(3,:) = ones(1,length(P(1,:)));
% A = hardlims(W*P)