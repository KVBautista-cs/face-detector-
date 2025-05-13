
imds = imageDatastore('C:\Users\Kevin\AppData\Local\Temp\6429c5c7-f4a2-400a-96d5-c3df87846f50_ECIS484-F24-Project4.zip.f50\ECIS484-F24-Project4', ...
                      'IncludeSubfolders', true, ...
                      'LabelSource', 'foldernames');
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

inputSize = [224 224];
augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augImdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

net = resnet50;
lgraph = layerGraph(net);

numClasses = numel(categories(imds.Labels)); 

newFc = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
                            'WeightLearnRateFactor', 10, ...
                            'BiasLearnRateFactor', 10);
newClassLayer = classificationLayer('Name', 'new_classoutput');


lgraph = replaceLayer(lgraph, 'fc1000', newFc);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassLayer);




options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 3, ...  
    'MiniBatchSize', 16, ...
    'ValidationData', augImdsValidation, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

myNet = trainNetwork(augImdsTrain, lgraph, options);
save myNet 

% testing with webcam
clc; clear all; close all;
cam = webcam;
load myNet; 
faceDetector = vision.CascadeObjectDetector;

while true
    img = snapshot(cam); 
    bboxes = step(faceDetector, img); 
    if ~isempty(bboxes)
        faceImg = imresize(imcrop(img, bboxes(1, :)), [224, 224]);
        
        label = classify(myNet, faceImg);
        
        img = insertObjectAnnotation(img, 'rectangle', bboxes, char(label));
        imshow(img);
    else
        imshow(img);
        title('No Face Detected');
    end
    drawnow;
end
