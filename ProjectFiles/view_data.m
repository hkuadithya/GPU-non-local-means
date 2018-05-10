
gpu = fread(fopen('output_gpu.bin', 'r'), 'float');

cpu = fread(fopen('output_cpu.bin', 'r'), 'float');


gpu = reshape(gpu, [181 217 51]);

cpu = reshape(cpu, [181 217 51]);


imtool(ground(:, :, 10), []);
imtool(noisy(:, :, 10), []);
imtool(cpu(:, :, 10), []);
imtool(gpu(:, :, 10), []);

