% this script helps in getting the surface normals for
% the videos of NYU depth dataset--
clc; clear all;

% addpath to the required 
addpath(genpath('./toolbox/'));
addpath('./utils/');

% ScanNet RGBD dataset
data_path = '/home/marsyang/Documents/Dataset/scannet/scans';

d = dir(data_path);
isub = [d(:).isdir]; %# returns logical vector
nameFolds = {d(isub).name}';
nameFolds(ismember(nameFolds,{'.','..'})) = [];

for i = 1 : size(nameFolds,1)
    subdir = fullfile(data_path,nameFolds{i},'extracted');
    disp(subdir)
    
    num_scans = size(dir(subdir),1)-3;
         
    for im = 1: num_scans
        tic;
        img_name =  fullfile(subdir, sprintf("frame-%.6d",im));
        % read the jth-depth image --
        imgDepth2 = im2double(imread(img_name+'.depth.pgm'));
        %imgDepth2 = double(swapbytes(imgDepth2));
        imgRgb2 = imread(img_name+'.color.jpg');

        % get the projection mask for NYUD2
        projectionSize = size(imgDepth2);

        % once the points are aligned (say according to RGB perspective)
        % compute 3D points.
        points3d = rgb_plane2rgb_world(imgDepth2);

        % using 3D points, compute surface normals --
        X = points3d(:,1);
        Y = -points3d(:,2);
        Z = points3d(:,3);
        [imgPlanes, imgNormals, normalConf, NCompute] = ...
            compute_local_planes(X, Y, Z, projectionSize);

        NMask = sum(NCompute.^2,3).^0.5 > 0.5;

        % tv-denoise the surface normals
        % this is same as David Fouhey's stuff.
        Ndash  = tvNormal(NCompute,1);
        N1 = zeros(size(imgRgb2,1), size(imgRgb2,2));
        N1 = Ndash(:,:,1);

        N2 = zeros(size(imgRgb2,1), size(imgRgb2,2));
        N2 = Ndash(:,:,2);

        N3 = zeros(size(imgRgb2,1), size(imgRgb2,2));
        N3 = Ndash(:,:,3);

        N = cat(3, N1, N2, N3);
        nx = N(:,:,1); ny = N(:,:,2); nz = N(:,:,3);
        Nn = (nx.^2 + ny.^2 + nz.^2).^0.5 + eps;
        nx = nx./Nn; ny = ny./Nn; nz = nz./Nn;

        % create visualization --
        Nvis =  uint8(255*(max(min(cat(3,nx,ny,nz),1),-1)+1)/2);
        Nx = Nvis(:,:,1);
        Ny = Nvis(:,:,2);
        Nz = Nvis(:,:,3);
        Nvis = uint8(cat(3,Nx, Ny, Nz));

        %imshow(Nvis)
        imwrite(Nvis,  img_name+'.normal.png');
        toc;
        disp(img_name+'.normal.png')

    end
end
