
clc;
clear;

% Load the ms_lesions mat file
load('ms_lesions.mat');

% Select 50 slices from the mat file
ground = ms_lesions(:, :, 50:100);

size(ground)

% These are the default parameters to the program
sigma = 10;
f = 1;
t = 5;

% This will Add white Gaussian noise to the ground truth
noisy = ground + sigma*randn(size(ground));

% The array is padded
padNoisy = padarray(noisy, [f f f], 'symmetric');


% Binary data is written to input.bin which is processed by the CUDA and C++ program.
fwrite(fopen('input.bin', 'w'), padNoisy, 'float');