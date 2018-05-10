function [ mse, psnr ] = mse_psnr(ground, noisy)
%MES Summary of this function goes here
%   Detailed explanation goes here
    
    ground = ground(:);
    noisy = noisy(:);
    
    mse = ((ground - noisy)' * (ground - noisy)) / numel(ground);
    
    psnr = 10 * log10(255 * 255 / mse);
    
end

