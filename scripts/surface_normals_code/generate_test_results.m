clear; clc; close all;

result_path = '../../src/outputs/test/';

image_number = 100;
image_size = 224;

for i = 1:image_number
    
    a = (1-cos(i/180*pi))*ones(image_size^2,1);
    dlmwrite(result_path+string(i)+'_norm_diff.txt',a);
    
end