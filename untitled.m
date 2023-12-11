% Turn off all warnings
warning('off','all')

% Add the path to the Matconvnet Mex file
addpath('.mex -setup\matconvnet-1.0-beta25\matlab\mex'); 

% Add path to utilities folder
addpath('.\utilities');

% Specify the test folder and network configurations
folderTest  = 'untitled';
networkTest = {'MS1', 'MS2', 'MS3'};
showResult  = 0;
writeRecon  = 1;
featureSize = 64;
blkSize     = 32; 
isLearnMtx  = [1, 0];
network     = networkTest{3}; 

% Loop through different sampling rates
for samplingRate = [0.1:0.1:0.3]
    % Load the pre-trained model
    modelName   = [network '_r' num2str(samplingRate)];
    data = load(fullfile('models', network ,[modelName,'.mat']));
    net  = dagnn.DagNN.loadobj(data.net);
    
    % Adjust the network for CSNet
    if strcmp(network,'CSNet')
        net.renameVar('x0', 'input'); 
        net.renameVar('x12', 'prediction'); 
    else
        net.removeLayer(net.layers(end).name) ;
    end
        
    % Set the network mode to test and move it to CPU
    net.mode = 'test';
    net.move('cpu');
        
    % Read images from the test set
    ext         =  {'*.jpg','*.png','*.bmp', '*.pgm', '*.tif'};
    filePaths   =  [];
    for i = 1 : length(ext)
        filePaths = cat(1,filePaths, dir(fullfile('testsets',folderTest,ext{i})));
    end
    
    % Initialize arrays for storing results
    PSNRs_CSNet = zeros(1,length(filePaths));
    SSIMs_CSNet = zeros(1,length(filePaths));
    time = zeros(1,length(filePaths));
    
    count = 1;
    allName = cell(1);
    
    % Process each image in the test set
    for i = 1:length(filePaths)
        % Read the image
        image = imread(fullfile('testsets', folderTest, filePaths(i).name));
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        allName{count} = nameCur;
        
        % Preprocess the image if it is RGB
        if size(image,3) == 3
            image = modcrop(image,32);
            % Remove the YCbCr conversion
            % image = rgb2ycbcr(image);
            % image = image(:,:,1);
        end
        label = im2single(image);
        
        % Check if the image size is a multiple of blkSize
        if mod(size(label, 1), blkSize) ~= 0 || mod(size(label, 2), blkSize) ~= 0
            continue
        end
        
        % Prepare the input data
        input = label;
        input = single(input);
        
        % Evaluate the network
        tic
        net.eval({'input', input});
        time(i) = toc; 
        out1 = net.getVarIndex('prediction') ;
        output = gather(squeeze(gather(net.vars(out1).value)));
        
        % Calculate PSNR and SSIM
        [PSNRCur_CSNet, SSIMCur_CSNet] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
        
        % Display results if required
        if showResult
            display(['        ' filePaths(i).name,'        ',num2str(PSNRCur_CSNet,'%2.2f'),'dB','    ',num2str(SSIMCur_CSNet,'%2.3f')])
        end
        
        % Store results
        PSNRs_CSNet(i) = PSNRCur_CSNet;
        SSIMs_CSNet(i) = SSIMCur_CSNet;
        
        % Save reconstructed images if required
        if writeRecon
            folder  = ['Results\2Image_' network ];
            if ~exist(folder), mkdir(folder); end
            fileName = [folder '\' folderTest '_' allName{count} '_subrate' num2str(samplingRate) '.png'];
            imwrite(im2uint8(output), fileName );
            
            count = count + 1;
        end
    end

    
    % Save results to a text file
    folder  = ['Results\1Text_' network ];
    if ~exist(folder), mkdir(folder); end
    imgName = [folderTest ];
    fileName = [folder '\' imgName '_subrate' num2str(samplingRate) '.txt'];
    write_txt(fileName, allName, samplingRate, PSNRs_CSNet, SSIMs_CSNet, time);
    
    % Display average results
    disp(['Average, subrate ' num2str(samplingRate) ': ' num2str(mean(PSNRs_CSNet), ...
           '%2.3f') 'dB, SSIM: ', num2str(mean(SSIMs_CSNet), '%2.4f'), ', time: ', num2str(mean(time), '%2.4f')]);
end
