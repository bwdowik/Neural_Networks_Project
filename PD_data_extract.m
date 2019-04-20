% PD_data_extract.m
%
% Description: Isolates parameters extracted from voice recordings of
% several patients with and without Parkinson's Disease (PD) into training
% and testing sets.
%				
% Variables:
%   1 - MDVP:Fo(Hz)
%   2 - MDVP:Fhi(Hz)
%   3 - MDVP:Flo(Hz)
%   4 - MDVP:Jitter(%)
%   5 - MDVP:Jitter(Abs)
%   6 - MDVP:RAP
%   7 - MDVP:PPQ
%   8 - Jitter:DDP
%   9 - MDVP:Shimmer
%   10 - MDVP:Shimmer(dB)
%   11 - Shimmer:APQ3
%   12 - Shimmer:APQ5
%   13 - MDVP:APQ
%   14 - Shimmer:DDA
%   15 - NHR
%   16 - HNR
%   17 - RPDE
%   18 - DFA
%   19 - spread1
%   20 - spread2
%   21 - D2
%   22 - PPE
%
% Authors: Braeden Benedict & Ben Wdowik
% Course: EE47058 (Neural Networks), University of Notre Dame
% Date: April 2019

%% Clear
clear
clc

%% Initialize Variables
n_patients_nPD = 8; %no PD (i.e. "healthy")
n_patients_yPD = 23; %yes PD
n_patients_tot = n_patients_nPD + n_patients_yPD; %total patients

%% Import Data
%import
data_file = 'parkinsons_data.csv';
PD_data_unsorted = readtable(data_file);

%reorder data by status
PD_data_sorted = sortrows(PD_data_unsorted,'status');
PD_data_status = table2array(PD_data_sorted(:,18)); %status variable

%% Determine Range for Healthy/PD and for Testing/Training
%find index range for healthy/PD
range_nPD = [1 (find(PD_data_status,1) - 1)]; %0 to last 0 before first 1
range_yPD = [find(PD_data_status,1) length(PD_data_status)]; %first 1 to end

%data range for training (cut in half)
train_range_nPD = [range_nPD(1) ceil(range_nPD(2)/2)];
train_range_yPD = [range_yPD(1) (ceil((range_yPD(2)-range_yPD(1))/2))+range_yPD(1)];

%data range for testing (leftover data)
test_range_nPD = [train_range_nPD(2)+1 range_nPD(2)];
test_range_yPD = [train_range_yPD(2)+1 range_yPD(2)];

%% Isolate Training and Testing Sets
%first variable (recording name)
PD_data_names = table2array(PD_data_sorted(:,1));

train_data_names = [PD_data_names(train_range_nPD(1):train_range_nPD(2));...
                        PD_data_names(train_range_yPD(1):train_range_yPD(2))];   
                    
test_data_names = [PD_data_names(test_range_nPD(1):test_range_nPD(2));...
                        PD_data_names(test_range_yPD(1):test_range_yPD(2))];
                    
%output variable (status) 
train_data_out = [PD_data_status(train_range_nPD(1):train_range_nPD(2));...
                        PD_data_status(train_range_yPD(1):train_range_yPD(2))];   
                    
test_data_out = [PD_data_status(test_range_nPD(1):test_range_nPD(2));...
                        PD_data_status(test_range_yPD(1):test_range_yPD(2))];

%all other 22 variables
for i=2:width(PD_data_sorted) %skip first variable (recording name)
    if i ~= 18 %skip status (i.e. output) variable
        if i<18 %re-index to account for skipped variable
            j = i-1;
        else
            j = i - 2;
        end
        
        %isolate variable
        PD_data_var(:,j) = table2array(PD_data_sorted(:,i));
        
        %isolate training data
        train_data_in(:,j) = [PD_data_var(train_range_nPD(1):train_range_nPD(2),j);...
                        PD_data_var(train_range_yPD(1):train_range_yPD(2),j)];
                  
        %isolate testing data
        test_data_in(:,j) = [PD_data_var(test_range_nPD(1):test_range_nPD(2),j);...
                        PD_data_var(test_range_yPD(1):test_range_yPD(2),j)];
    end
end
