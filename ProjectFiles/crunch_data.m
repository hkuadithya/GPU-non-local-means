clear;
clc;
close all;
hold off;

data_transform;

%files = {'output_cpu_t3', 'output_cpu_t4', 'output_cpu_t5', 'output_cpu_t6', 'output_cpu_t7', 'output_cpu_t8',...
%    'output_gpu_t3', 'output_gpu_t4', 'output_gpu_t5', 'output_gpu_t6', 'output_gpu_t7', 'output_gpu_t8' };

files = {'output_NORM_1', 'output_NORM_2', 'output_NORM_3', 'output_NORM_4'};

[mse, psnr] = mse_psnr(ground, noisy);
fprintf('noisy\tmse = %f\tpsnr = %f\t\n', mse, psnr);

for file=files
    
    data = fread(fopen(char(file), 'r'), 'float');
    
    data = reshape(data, [181, 217, 51]);
    
    [mse, psnr] = mse_psnr(ground, data);
    
    imtool(data(:, :, 1), []);
    
    fprintf('%s\tmse = %f\tpsnr = %f\t\n',char(file), mse, psnr);
    
end;


%{
xtick = {'7^3' ; '9^3' ; '11^3' ; '13^3' ; '15^3' ; '17^3'};

y1 = [51.19, 104.57, 188.19, 308.29, 468.16, 676.29];

y2 = [2.07, 4.35, 7.87, 12.98, 19.76, 28.39];

plot(y1, 'r*-');

hold on;
grid on;

plot(y2, 'b*-');

set(gca,'xtick',(1:6),'xticklabel',xtick);

title('Serial vs. Parallel NL Means exec. time');

xlabel('Search window dimensions');

ylabel('Execution time (seconds)');

legend('Conventional NL Means', 'GPU Accelerated NL Means');
%}

%{
load benchmark.txt;
X = int32(benchmark(:, 1:3));
Y = benchmark(:, 4);

x = X(:, 1);                                        % longitude data
y = X(:, 2);                                        % latitude data
z = X(:, 3);                                        % percent rural data

execution_time = Y;                                 % fatalities data

scatter3(x,y,z,15,execution_time,'filled')    % draw the scatter plot
ax = gca;
%ax.XDir = 'reverse';
%view(-31,14);
xlabel('x-dimension');
ylabel('y-dimension');
zlabel('z-dimension');
title('Exec. time for 119 (x, y, z) launch configurations');
cb = colorbar;                                     % create and label the colorbar
cb.Label.String = 'Execution Time(seconds)';
colormap jet;

%}



