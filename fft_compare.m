clear;

file1 = 'res_top_fft.cu_conv1.txt@';
file2 = 'res_top_fft.cpp_conv1.txt';

id_gpu = fopen(file1);
tmp = textscan(id_gpu, '%f + %f * i');
gpu = complex(tmp{1,1},tmp{1,2});

id_cpu = fopen(file2);
tmp = textscan(id_cpu, '%f + %f * i');
cpu = complex(tmp{1,1},tmp{1,2});

diff = cpu - gpu;
diff_max = max(diff);