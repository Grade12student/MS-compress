function [imdb] = generatepatches

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @article{zhang2017beyond,
%   title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
%   author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
%   journal={IEEE Transactions on Image Processing},
%   year={2017},
%   volume={26},
%   number={7},
%   pages={3142-3155},
% }

% by Kai Zhang (1/2018)
% cskaizhang@gmail.com
% https://github.com/cszn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('utilities');
batchSize     = 64;        % batch size
%folder        = 'D:\Github\DnCNN_Org\TrainingCodes\DnCNN_TrainingCodes_v1.0\data\Train400';  %
folder        = 'D:\Github\TrainingImage';
imgSet        = {'BSDS_All' 'DIV2K_train_HR' 'DIV2K_test_HR' 'pristine_images'};
train_id      = [1 ];
nchannel      = 1;          % number of channels
patchsize     = 224;

stride        = 16;

step1         = randi(stride)-1;
step2         = randi(stride)-1;
count         = 0;
ext           =  {'*.jpg','*.png','*.bmp','*.jpeg'};
filepaths     =  [];

for i = 1 : length(ext)
    for j = 1:1:length(train_id)
        filepaths = cat(1,filepaths, dir(fullfile(folder, imgSet{train_id(j)}, ext{i})));
    end
end

% count the number of extracted patches
scales  = [1 0.9 0.8 0.7]; % scale the image to augment the training data

for i = 1 : length(filepaths)    
    imgRGB = imread(fullfile( filepaths(i).folder, filepaths(i).name)); % uint8
    for j = 1:1:1
        im0 =rgb2gray( imgRGB);
        
        % [~, name, exte] = fileparts(filepaths(i).name);
        if mod(i,100)==0
            disp([i,length(filepaths)]);
        end
        for s = 1:4
            im = imresize(im0,scales(s),'bicubic');
            [hei,wid,~] = size(im);
            for x = 1+step1 : stride : (hei-patchsize+1)
                for y = 1+step2 :stride : (wid-patchsize+1)
                    count = count+1;
                end
            end
        end
    end
end

numPatches  = ceil(count/batchSize)*batchSize;
diffPatches = numPatches - count;
disp([int2str(numPatches),' = ',int2str(numPatches/batchSize),' X ', int2str(batchSize)]);


count = 0;
imdb.labels  = zeros(patchsize, patchsize, nchannel, numPatches,'single');

for i = 1 : length(filepaths)
    
    imgRGB = imread(fullfile( filepaths(i).folder,filepaths(i).name)); % uint8
    %[~, name, exte] = fileparts(filepaths(i).name);
    for j = 1:1:1
        im0 = rgb2gray(imgRGB);
        
        if mod(i,100)==0
            disp([i,length(filepaths)]);
        end
        for s = 1:4
            im = imresize(im0,scales(s),'bicubic');
            for j = 1:1
                image_aug   = data_augmentation(im, j);  % augment data
                im_label    = im2single(image_aug);         % single
                [hei,wid,~] = size(im_label);
                
                for x = 1+step1 : stride : (hei-patchsize+1)
                    for y = 1+step2 :stride : (wid-patchsize+1)
                        count       = count+1;
                        imdb.labels(:, :, :, count)   = im_label(x : x+patchsize-1, y : y+patchsize-1,:);
                        if count<=diffPatches
                            imdb.labels(:, :, :, end-count+1)   = im_label(x : x+patchsize-1, y : y+patchsize-1,:);
                        end
                    end
                end
            end
        end
    end
end

imdb.set    = uint8(ones(1,size(imdb.labels,4)));

