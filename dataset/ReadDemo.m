clc
clear

%load annotation
load('annotation_1.mat');

%Set data root
DataRoot='./DataSet_001/';

for i=1:numel(annot)
    %Read image
    im=imread([DataRoot,annot(i).ImgName]);
    %Show image
    imshow(im);
    hold on
    %Plot bounding box
    for j=1:size(annot(i).BBox,1)
        x1=annot(i).BBox(j,1);
        y1=annot(i).BBox(j,2);
        x2=annot(i).BBox(j,3);
        y2=annot(i).BBox(j,4);
        plot([x1,x2],[y1,y1],'r','LineWidth',2);
        hold on
        plot([x1,x2],[y2,y2],'r','LineWidth',2);
        hold on
        plot([x1,x1],[y1,y2],'r','LineWidth',2);
        hold on
        plot([x2,x2],[y1,y2],'r','LineWidth',2);
        hold on
    end
    pause(0.00001);
end