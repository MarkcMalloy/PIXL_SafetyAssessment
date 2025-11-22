function[header,im, metaData, metaDataBin]=readAscImage(file,readMeta)
% Only unc and jp0  and cen format implemented

% 
% if(nargin < 2)
%     % Take image format from file name if not given as input
%     [a, b, imageFormat] = fileparts(file);
%     % And remove the '.'
%     imageFormat(1) = [];
% end
% Example run cmd: [header, im, metaData, metaDataBin] = readAscImage('A250922_11505743.unc')
% [header, im, metaData, metaDataBin] = readAscImage('pixl_7.unc')
% I8 = uint8(255 * mat2gray(im))
% imwrite(I8, '40mm_7.png')
s = dir(file);
if isempty(s)
    header = [];    im = [];    metaData = []; metaDataBin = [];
    return 
end
fileSize = s.bytes;
fid=fopen(file);





%% Read header
temp = fread(fid,2,'uint16');

k = 0;
if temp(1) == 65535
    k = k+1;
    %compensate for hexapod metadata
    temp = fread(fid,2*7,'uint16');
    temp = fread(fid,2,'uint16');
end


header.DAC_OFFFSET=temp(1);
header.DAC_gain=temp(2);

temp = fread(fid,6,'uint8');
header.INTEGRATION_TIME=temp(1);
header.COMPRESSION=temp(2);
header.ROI=temp(3);
header.JPEG_QUALITY=temp(4);
header.COMPRESSION_THRESHOLD=temp(5);
header.INFO=temp(6);

temp = fread(fid,2,'uint16');
header.VALID=temp(1);
header.STATUS=temp(2);

temp = fread(fid,2,'uint32');
header.CODE_START=temp(1);
header.CODE_END=temp(2);

temp = fread(fid,1,'uint16');
header.SUB_TIMESTAMP=temp(1);

temp = fread(fid,1,'uint32');
header.TIMESTAMP=temp(1);

temp = fread(fid,3,'uint16');
header.H=temp(1);
header.W=temp(2);
header.IMOD=temp(3);


%% Read image data


if header.COMPRESSION == 0
    imageFormat = 'unc';
elseif header.COMPRESSION == 1
    imageFormat = 'cen';
elseif header.COMPRESSION == 2
    imageFormat = 'roi';
elseif header.COMPRESSION == 3
    imageFormat = 'jp0';
elseif header.COMPRESSION == 4
    imageFormat = 'nonStlOb';
elseif header.COMPRESSION == 32
    imageFormat = 'CC';
elseif header.COMPRESSION == 33
    imageFormat = 'TRN';
elseif header.COMPRESSION == 34    
    imageFormat = 'sli';
end

trn = 0;
switch imageFormat
    case 'unc'
        n=header.H*(header.IMOD+1)*header.W;
        temp = fread(fid,n,'*uint8');
        if (temp(1) == 36 && temp(2) == 84 && temp(3) == 70 && temp(4) == 36) % $TF$ 
            trn = 1;
        else
            im=reshape(temp, header.W, header.H*(header.IMOD+1))';
        end
    case 'jp0'
%         n=header.H*(header.IMOD+1)*header.W;
        n = (header.CODE_END-header.CODE_START);
        temp = fread(fid,n,'*uint8');
        % Write the binary data to a temporary file
        tname = tempname;
        fidTemp = fopen(tname, 'wb');
        fwrite(fidTemp, temp);
        fclose(fidTemp);
        % Load the file as a jpeg file
        im=imread(tname, 'jpg');
        % And delete the temporary file
        delete(tname);
    case 'cen'        
%         n = (-34)/10;
        n = (header.CODE_END-header.CODE_START)/10;
        im = zeros(n,3);
        for i = 1:n
            im(i,1) = double(fread(fid,1,'*uint32'))/5000000;
            im(i,2) = double(fread(fid,1,'*uint32'))/5000000;
            im(i,3) = double(fread(fid,1,'*uint16'));
        end
    case 'roi'
        n = (header.CODE_END-header.CODE_START)/10; 
        i = 1;
        
        while(1)
            roix = double(fread(fid,1,'*uint16'));
            roiy = double(fread(fid,1,'*uint16'));
            roiw = double(fread(fid,1,'*uint16'));
            roih = double(fread(fid,1,'*uint16'));
            temp = double(fread(fid,roiw*roih,'*uint8'));
            roi = reshape(temp, roiw, roih)';
            roiSize = size(roi,1)*size(roi,2);
            im{i} = struct('x', roix, 'y', roiy, 'w', roiw, 'h', roih, 'roi', roi);
%             if roiSize > im(3)*im(4)
%                 im = [roix; roiy; roiw; roih; roi(:)];
%             end
            count = ftell(fid);
            i = i+1;
            if count == header.CODE_END
                
                break
                
            end
        end
    case 'sli'
        im.sliHeader.Ncen = fread(fid,1,'*uint8');
        im.sliHeader.Nmatch = fread(fid,1,'*uint8');
        im.sliHeader.RMSE = fread(fid,1,'*uint8')/10;
        temp = fread(fid,1,'*uint8');
        im.sliHeader.planeN(1) = double(fread(fid,1,'*int32'))/2147483647;
        im.sliHeader.planeN(2) = double(fread(fid,1,'*int32'))/2147483647;
        im.sliHeader.planeN(3) = double(fread(fid,1,'*int32'))/2147483647;
        im.sliHeader.planeD = fread(fid,1,'*uint32');
        im.sliHeader.planeRMSE = fread(fid,1,'*uint16');
        temp = fread(fid,1,'*uint16');
        im.sliHeader.warning = fread(fid,1,'*uint8');
        im.sliHeader.error = fread(fid,1,'*uint8');
        for i=1:im.sliHeader.Ncen
            im.sliData.cen(i,1) = double(fread(fid,1,'*uint32'))/5000000;
            im.sliData.cen(i,2) = double(fread(fid,1,'*uint32'))/5000000;
            im.sliData.cen(i,3) = double(fread(fid,1,'*uint16'));
            im.sliData.Q(i,1) = double(fread(fid,1,'*int32'));
            im.sliData.Q(i,2) = double(fread(fid,1,'*int32'));
            im.sliData.Q(i,3) = double(fread(fid,1,'*int32'));
            im.sliData.sliID(i,1) = double(fread(fid,1,'*uint8'));
            im.sliData.residual(i,1) = double(fread(fid,1,'*uint8'))/10;
        end
    otherwise
        im=0;
end

if trn
    
    %feat set header
    id = 5;
    im.TRNRefHeader.numFeat = temp(id) + temp(id+1)*2^8;
    id = id+2;
    s = strcat(dec2hex(temp(id+7),2), dec2hex(temp(id+6),2), dec2hex(temp(id+5),2), dec2hex(temp(id+4),2),dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));    
    im.TRNRefHeader.solT(1) = typecast(uint64(hex2dec(s)),'double');
    id = id+8;
    s = strcat(dec2hex(temp(id+7),2), dec2hex(temp(id+6),2), dec2hex(temp(id+5),2), dec2hex(temp(id+4),2),dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));    
    im.TRNRefHeader.solT(2) = typecast(uint64(hex2dec(s)),'double');
    id = id+8;
    s = strcat(dec2hex(temp(id+7),2), dec2hex(temp(id+6),2), dec2hex(temp(id+5),2), dec2hex(temp(id+4),2),dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));    
    im.TRNRefHeader.solT(3) = typecast(uint64(hex2dec(s)),'double');
    id = id+8;
    s = strcat(dec2hex(temp(id+3)), dec2hex(temp(id+2)), dec2hex(temp(id+1)), dec2hex(temp(id)));    
    im.TRNRefHeader.planeN(1) = typecast(uint32(hex2dec(s)),'single');
    id = id+4;
    s = strcat(dec2hex(temp(id+3)), dec2hex(temp(id+2)), dec2hex(temp(id+1)), dec2hex(temp(id)));    
    im.TRNRefHeader.planeN(2) = typecast(uint32(hex2dec(s)),'single');
    id = id+4;
    s = strcat(dec2hex(temp(id+3)), dec2hex(temp(id+2)), dec2hex(temp(id+1)), dec2hex(temp(id)));    
    im.TRNRefHeader.planeN(3) = typecast(uint32(hex2dec(s)),'single');
    id = id+4;
    s = strcat(dec2hex(temp(id+3)), dec2hex(temp(id+2)), dec2hex(temp(id+1)), dec2hex(temp(id)));    
    im.TRNRefHeader.planeD = typecast(uint32(hex2dec(s)),'single');
    id = id+4;
    
    im.TRNRefHeader.timeCoarse = 0;
    im.TRNRefHeader.timeFine = 0;
    id = id+6;
    im.TRNRefHeader.RangeSetBySLI = temp(id);
    id = id+1;
    im.TRNRefHeader.RefID = temp(id);
    id = id+1 + 10;
    %spare 10
    offset = id;
    featSize = 52;
    for i = 0:double(im.TRNRefHeader.numFeat-1)
        id = (offset + (i)*featSize);
        im.TRNRefSet(i+1).p(1) = hex2dec(strcat(dec2hex(temp(id+1),2), dec2hex(temp(id),2)));
        id = (offset + (i)*featSize)+2;
        im.TRNRefSet(i+1).p(2) = hex2dec(strcat(dec2hex(temp(id+1),2), dec2hex(temp(id),2)));
        id = (offset + (i)*featSize)+4;
        s = strcat(dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));            
        im.TRNRefSet(i+1).q(1) = typecast(uint32(hex2dec(s)),'single');
        id = (offset + (i)*featSize)+8;
        s = strcat(dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));            
        im.TRNRefSet(i+1).q(2) = typecast(uint32(hex2dec(s)),'single');
        id = (offset + (i)*featSize)+12;
        s = strcat(dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));            
        im.TRNRefSet(i+1).q(3) = typecast(uint32(hex2dec(s)),'single');
        id = (offset + (i)*featSize)+16;
        im.TRNRefSet(i+1).metric = hex2dec(strcat(dec2hex(temp(id+1),2), dec2hex(temp(id),2)));
    end
    
    
    %feat set header
    id = (offset + (i+1)*featSize)+4;
    im.TRNCurrHeader.numFeat = temp(id) + temp(id+1)*2^8;
    id = id+2;
    s = strcat(dec2hex(temp(id+7),2), dec2hex(temp(id+6),2), dec2hex(temp(id+5),2), dec2hex(temp(id+4),2),dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));    
    im.TRNCurrHeader.solT(1) = typecast(uint64(hex2dec(s)),'double');
    id = id+8;
    s = strcat(dec2hex(temp(id+7),2), dec2hex(temp(id+6),2), dec2hex(temp(id+5),2), dec2hex(temp(id+4),2),dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));    
    im.TRNCurrHeader.solT(2) = typecast(uint64(hex2dec(s)),'double');
    id = id+8;
    s = strcat(dec2hex(temp(id+7),2), dec2hex(temp(id+6),2), dec2hex(temp(id+5),2), dec2hex(temp(id+4),2),dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));    
    im.TRNCurrHeader.solT(3) = typecast(uint64(hex2dec(s)),'double');
    id = id+8;
    s = strcat(dec2hex(temp(id+3)), dec2hex(temp(id+2)), dec2hex(temp(id+1)), dec2hex(temp(id)));    
    im.TRNCurrHeader.planeN(1) = typecast(uint32(hex2dec(s)),'single');
    id = id+4;
    s = strcat(dec2hex(temp(id+3)), dec2hex(temp(id+2)), dec2hex(temp(id+1)), dec2hex(temp(id)));    
    im.TRNCurrHeader.planeN(2) = typecast(uint32(hex2dec(s)),'single');
    id = id+4;
    s = strcat(dec2hex(temp(id+3)), dec2hex(temp(id+2)), dec2hex(temp(id+1)), dec2hex(temp(id)));    
    im.TRNCurrHeader.planeN(3) = typecast(uint32(hex2dec(s)),'single');
    id = id+4;
    s = strcat(dec2hex(temp(id+3)), dec2hex(temp(id+2)), dec2hex(temp(id+1)), dec2hex(temp(id)));    
    im.TRNCurrHeader.planeD = typecast(uint32(hex2dec(s)),'single');
    id = id+4;
    
    im.TRNCurrHeader.timeCoarse = 0;
    im.TRNCurrHeader.timeFine = 0;
    id = id+6;
    im.TRNCurrHeader.RangeSetBySLI = temp(id);
    id = id+1;
    im.TRNCurrHeader.RefID = temp(id);
    id = id+1 + 10;
    %spare 10
    offset = id;
    featSize = 52;
    for i = 0:double(im.TRNCurrHeader.numFeat-1)
        id = (offset + (i)*featSize);
        im.TRNCurrSet(i+1).p(1) = hex2dec(strcat(dec2hex(temp(id+1),2), dec2hex(temp(id),2)));
        id = (offset + (i)*featSize)+2;
        im.TRNCurrSet(i+1).p(2) = hex2dec(strcat(dec2hex(temp(id+1),2), dec2hex(temp(id),2)));
        id = (offset + (i)*featSize)+4;
        s = strcat(dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));            
        im.TRNCurrSet(i+1).q(1) = typecast(uint32(hex2dec(s)),'single');
        id = (offset + (i)*featSize)+8;
        s = strcat(dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));            
        im.TRNCurrSet(i+1).q(2) = typecast(uint32(hex2dec(s)),'single');
        id = (offset + (i)*featSize)+12;
        s = strcat(dec2hex(temp(id+3),2), dec2hex(temp(id+2),2), dec2hex(temp(id+1),2), dec2hex(temp(id),2));            
        im.TRNCurrSet(i+1).q(3) = typecast(uint32(hex2dec(s)),'single');
        id = (offset + (i)*featSize)+16;
        im.TRNCurrSet(i+1).metric = hex2dec(strcat(dec2hex(temp(id+1),2), dec2hex(temp(id),2)));
    end
end

%% Read metaData
metaData = [];
metaDataBin = [];
if nargin == 1
    readMeta = 1;
else
    readMeta = 0;
end



if readMeta
    %check if metadata is available
    temp = fread(fid,4,'*uint8');
    if header.INFO >= 128  & ~isempty(temp)
        %read until metadata signal start
        
        while 1
            if temp(1) == 36 && temp(2) == 77 && temp(3) == 68 && temp(4) == 36 % $MD$ 
                break
            else
                temp(1) = temp(2);
                temp(2) = temp(3);
                temp(3) = temp(4);
                temp(4) = fread(fid,1,'*uint8');
            end
        end
        metaDataLength = fread(fid,1,'*uint32');
        metaDataN = metaDataLength/6;
        %Read meta data
        for i = 1:metaDataN

            %read meta data type
            metaDataType = fread(fid,1,'*uint16');

            %read associated meta data
            switch metaDataType
                case hex2dec('0000')    %   SHUTTIME
                    temp = fread(fid,1,'*uint32');
                    metaData.shutterTime = temp;
                    metaDataBin.shutterTime = dec2hex(temp,8);
                    
                case hex2dec('0100')    %   SLICFG_A
                    temp = fread(fid,1,'*uint16');
                    metaData.SLIAconf_currLvl = bitand(temp, 511, 'uint16')*3;  %mA
                    metaData.SLIAconf_capCharge = bitand(temp, 2^12, 'uint16')/2^12;
                    metaData.SLIAconf_sliEn = bitand(temp, 2^13, 'uint16')/2^13;
                    metaData.SLIAconf_flashCtrl = bitand(temp, 2^14, 'uint16')/2^14;
                    metaData.SLIAconf_xsubCtrl = bitand(temp, 2^15, 'uint16')/2^15;
                    
                    metaDataBin.SLIAconf_currLvl = dec2hex(bitand(temp, 511, 'uint16'), 4);  %mA
                    metaDataBin.SLIAconf_capCharge = dec2hex(bitand(temp, 2^12, 'uint16'), 4);
                    metaDataBin.SLIAconf_sliEn = dec2hex(bitand(temp, 2^13, 'uint16'), 4);
                    metaDataBin.SLIAconf_flashCtrl = dec2hex(bitand(temp, 2^14, 'uint16'), 4);
                    metaDataBin.SLIAconf_xsubCtrl = dec2hex(bitand(temp, 2^15, 'uint16'), 4);
                    
                    temp = fread(fid,1,'*uint16');
                    metaData.SLIAconf_flashT = double(temp)/(2*4.608e6);                 
                    metaDataBin.SLIAconf_flashT = dec2hex(temp, 4);
                    
                case hex2dec('0101')    %   SLICFG_B
                    temp = fread(fid,1,'*uint16');
                    metaData.SLIBconf_currLvl = bitand(temp, 511, 'uint16')*3;  %mA
                    metaData.SLIBconf_capCharge = bitand(temp, 2^12, 'uint16')/2^12;
                    metaData.SLIBconf_sliEn = bitand(temp, 2^13, 'uint16')/2^13;
                    metaData.SLIBconf_flashCtrl = bitand(temp, 2^14, 'uint16')/2^14;
                    metaData.SLIBconf_xsubCtrl = bitand(temp, 2^15, 'uint16')/2^15;
                    
                    metaDataBin.SLIBconf_currLvl = dec2hex(bitand(temp, 511, 'uint16'), 4);  %mA
                    metaDataBin.SLIBconf_capCharge = dec2hex(bitand(temp, 2^12, 'uint16'), 4);
                    metaDataBin.SLIBconf_sliEn = dec2hex(bitand(temp, 2^13, 'uint16'), 4);
                    metaDataBin.SLIBconf_flashCtrl = dec2hex(bitand(temp, 2^14, 'uint16'),4);
                    metaDataBin.SLIBconf_xsubCtrl = dec2hex(bitand(temp, 2^15, 'uint16'), 4);
                    
                    temp = fread(fid,1,'*uint16');
                    metaData.SLIBconf_flashT = double(temp)/(2*4.608e6);
                    metaDataBin.SLIBconf_flashT = dec2hex(temp, 4);
                    
                case hex2dec('0200')    %   SLIFLICL    
                    temp = fread(fid,1,'*uint32');
                    metaData.SLIFLIctrl_cycleType = bitand(temp, 2^0 + 2^1, 'uint32');
                    metaData.SLIFLIctrl_fliConf = bitand(temp, 2^2, 'uint32')/2^2;
                    metaData.SLIFLIctrl_fliHK = bitand(temp, 2^5, 'uint32')/2^5;
                    metaData.SLIFLIctrl_fliApower = bitand(temp, 2^6, 'uint32')/2^6;
                    metaData.SLIFLIctrl_fliBpower = bitand(temp, 2^7, 'uint32')/2^7;
                    metaData.SLIFLIctrl_fliAbank1 = bitand(temp, 2^16, 'uint32')/2^16;
                    metaData.SLIFLIctrl_fliAbank2 = bitand(temp, 2^17, 'uint32')/2^17;
                    metaData.SLIFLIctrl_fliAbank3 = bitand(temp, 2^18, 'uint32')/2^18;
                    metaData.SLIFLIctrl_fliAbank4 = bitand(temp, 2^19, 'uint32')/2^19;
                    metaData.SLIFLIctrl_fliAstrOdd = bitand(temp, 2^20, 'uint32')/2^20;
                    metaData.SLIFLIctrl_fliAstrEven = bitand(temp, 2^21, 'uint32')/2^21;
                    metaData.SLIFLIctrl_fliBbank1 = bitand(temp, 2^24, 'uint32')/2^24;
                    metaData.SLIFLIctrl_fliBbank2 = bitand(temp, 2^25, 'uint32')/2^25;
                    metaData.SLIFLIctrl_fliBbank3 = bitand(temp, 2^26, 'uint32')/2^26;
                    metaData.SLIFLIctrl_fliBbank4 = bitand(temp, 2^27, 'uint32')/2^27;
                    metaData.SLIFLIctrl_fliBstrOdd = bitand(temp, 2^28, 'uint32')/2^28;
                    metaData.SLIFLIctrl_fliBstrEven = bitand(temp, 2^29, 'uint32')/2^29;
                    
                    metaDataBin.SLIFLIctrl_cycleType = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliConf = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliHK = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliApower = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliBpower = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliAbank1 = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliAbank2 = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliAbank3 = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliAbank4 = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliAstrOdd = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliAstrEven = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliBbank1 = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliBbank2 = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliBbank3 = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliBbank4 = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliBstrOdd = dec2hex(temp, 8);
                    metaDataBin.SLIFLIctrl_fliBstrEven = dec2hex(temp, 8);
                case hex2dec('0300')    %   FLICFG_A
                    temp = fread(fid,1,'*uint32');
                    metaData.FLIAconf_flashCtrlStr1 = bitand(temp, 2^0, 'uint32')/2^0;
                    metaData.FLIAconf_flashCtrlStr2 = bitand(temp, 2^1, 'uint32')/2^1;
                    metaData.FLIAconf_flashCtrlStr3 = bitand(temp, 2^2, 'uint32')/2^2;
                    metaData.FLIAconf_flashCtrlStr4 = bitand(temp, 2^3, 'uint32')/2^3;
                    metaData.FLIAconf_flashCtrlStr5 = bitand(temp, 2^4, 'uint32')/2^4;
                    metaData.FLIAconf_flashCtrlStr6 = bitand(temp, 2^5, 'uint32')/2^5;
                    metaData.FLIAconf_flashCtrlStr7 = bitand(temp, 2^6, 'uint32')/2^6;
                    metaData.FLIAconf_flashCtrlStr8 = bitand(temp, 2^7, 'uint32')/2^7;
                    metaData.FLIAconf_xsubCtrlStr1 = bitand(temp, 2^8, 'uint32')/2^8;
                    metaData.FLIAconf_xsubCtrlStr2 = bitand(temp, 2^9, 'uint32')/2^9;
                    metaData.FLIAconf_xsubCtrlStr3 = bitand(temp, 2^10, 'uint32')/2^10;
                    metaData.FLIAconf_xsubCtrlStr4 = bitand(temp, 2^11, 'uint32')/2^11;
                    metaData.FLIAconf_xsubCtrlStr5 = bitand(temp, 2^12, 'uint32')/2^12;
                    metaData.FLIAconf_xsubCtrlStr6 = bitand(temp, 2^13, 'uint32')/2^13;
                    metaData.FLIAconf_xsubCtrlStr7 = bitand(temp, 2^14, 'uint32')/2^14;
                    metaData.FLIAconf_xsubCtrlStr8 = bitand(temp, 2^15, 'uint32')/2^15;
                    metaData.FLIAconf_currLvlStr1 = bitand(temp, 2^16 + 2^17, 'uint32')/(2^16)*140;
                    if metaData.FLIAconf_currLvlStr1 > 280,	metaData.FLIAconf_currLvlStr1 = 500;  end
                    metaData.FLIAconf_currLvlStr2 = bitand(temp, 2^18 + 2^19, 'uint32')/(2^18)*140;
                    if metaData.FLIAconf_currLvlStr2 > 280,	metaData.FLIAconf_currLvlStr2 = 500;  end
                    metaData.FLIAconf_currLvlStr3 = bitand(temp, 2^20 + 2^21, 'uint32')/(2^20)*140;
                    if metaData.FLIAconf_currLvlStr3 > 280,	metaData.FLIAconf_currLvlStr3 = 500;  end
                    metaData.FLIAconf_currLvlStr4 = bitand(temp, 2^22 + 2^23, 'uint32')/(2^22)*140;
                    if metaData.FLIAconf_currLvlStr4 > 280,	metaData.FLIAconf_currLvlStr4 = 500;  end
                    metaData.FLIAconf_currLvlStr5 = bitand(temp, 2^24 + 2^25, 'uint32')/(2^24)*140;
                    if metaData.FLIAconf_currLvlStr5 > 280,	metaData.FLIAconf_currLvlStr5 = 500;  end
                    metaData.FLIAconf_currLvlStr6 = bitand(temp, 2^26 + 2^27, 'uint32')/(2^26)*140;
                    if metaData.FLIAconf_currLvlStr6 > 280,	metaData.FLIAconf_currLvlStr6 = 500;  end
                    metaData.FLIAconf_currLvlStr7 = bitand(temp, 2^28 + 2^29, 'uint32')/(2^28)*140;
                    if metaData.FLIAconf_currLvlStr7 > 280,	metaData.FLIAconf_currLvlStr7 = 500;  end
                    metaData.FLIAconf_currLvlStr8 = bitand(temp, 2^30 + 2^31, 'uint32')/(2^30)*140;
                    if metaData.FLIAconf_currLvlStr8 > 280,	metaData.FLIAconf_currLvlStr8 = 500;  end
                    
                    metaDataBin.FLIAconf_flashCtrlStr1 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_flashCtrlStr2 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_flashCtrlStr3 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_flashCtrlStr4 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_flashCtrlStr5 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_flashCtrlStr6 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_flashCtrlStr7 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_flashCtrlStr8 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_xsubCtrlStr1 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_xsubCtrlStr2 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_xsubCtrlStr3 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_xsubCtrlStr4 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_xsubCtrlStr5 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_xsubCtrlStr6 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_xsubCtrlStr7 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_xsubCtrlStr8 = dec2hex(temp, 8);
                    metaDataBin.FLIAconf_currLvlStr1 = dec2hex(temp, 8);
                    
                case hex2dec('0301')    %   FLICFG_B
                    temp = fread(fid,1,'*uint32');
                    metaData.FLIBconf_flashCtrlStr1 = bitand(temp, 2^0, 'uint32')/2^0;
                    metaData.FLIBconf_flashCtrlStr2 = bitand(temp, 2^1, 'uint32')/2^1;
                    metaData.FLIBconf_flashCtrlStr3 = bitand(temp, 2^2, 'uint32')/2^2;
                    metaData.FLIBconf_flashCtrlStr4 = bitand(temp, 2^3, 'uint32')/2^3;
                    metaData.FLIBconf_flashCtrlStr5 = bitand(temp, 2^4, 'uint32')/2^4;
                    metaData.FLIBconf_flashCtrlStr6 = bitand(temp, 2^5, 'uint32')/2^5;
                    metaData.FLIBconf_flashCtrlStr7 = bitand(temp, 2^6, 'uint32')/2^6;
                    metaData.FLIBconf_flashCtrlStr8 = bitand(temp, 2^7, 'uint32')/2^7;
                    metaData.FLIBconf_xsubCtrlStr1 = bitand(temp, 2^8, 'uint32')/2^8;
                    metaData.FLIBconf_xsubCtrlStr2 = bitand(temp, 2^9, 'uint32')/2^9;
                    metaData.FLIBconf_xsubCtrlStr3 = bitand(temp, 2^10, 'uint32')/2^10;
                    metaData.FLIBconf_xsubCtrlStr4 = bitand(temp, 2^11, 'uint32')/2^11;
                    metaData.FLIBconf_xsubCtrlStr5 = bitand(temp, 2^12, 'uint32')/2^12;
                    metaData.FLIBconf_xsubCtrlStr6 = bitand(temp, 2^13, 'uint32')/2^13;
                    metaData.FLIBconf_xsubCtrlStr7 = bitand(temp, 2^14, 'uint32')/2^14;
                    metaData.FLIBconf_xsubCtrlStr8 = bitand(temp, 2^15, 'uint32')/2^15;
                    metaData.FLIBconf_currLvlStr1 = bitand(temp, 2^16 + 2^17, 'uint32')/(2^16)*140;
                    if metaData.FLIBconf_currLvlStr1 > 280,	metaData.FLIBconf_currLvlStr1 = 500;  end
                    metaData.FLIBconf_currLvlStr2 = bitand(temp, 2^18 + 2^19, 'uint32')/(2^18)*140;
                    if metaData.FLIBconf_currLvlStr2 > 280,	metaData.FLIBconf_currLvlStr2 = 500;  end
                    metaData.FLIBconf_currLvlStr3 = bitand(temp, 2^20 + 2^21, 'uint32')/(2^20)*140;
                    if metaData.FLIBconf_currLvlStr3 > 280,	metaData.FLIBconf_currLvlStr3 = 500;  end
                    metaData.FLIBconf_currLvlStr4 = bitand(temp, 2^22 + 2^23, 'uint32')/(2^22)*140;
                    if metaData.FLIBconf_currLvlStr4 > 280,	metaData.FLIBconf_currLvlStr4 = 500;  end
                    metaData.FLIBconf_currLvlStr5 = bitand(temp, 2^24 + 2^25, 'uint32')/(2^24)*140;
                    if metaData.FLIBconf_currLvlStr5 > 280,	metaData.FLIBconf_currLvlStr5 = 500;  end
                    metaData.FLIBconf_currLvlStr6 = bitand(temp, 2^26 + 2^27, 'uint32')/(2^26)*140;
                    if metaData.FLIBconf_currLvlStr6 > 280,	metaData.FLIBconf_currLvlStr6 = 500;  end
                    metaData.FLIBconf_currLvlStr7 = bitand(temp, 2^28 + 2^29, 'uint32')/(2^28)*140;
                    if metaData.FLIBconf_currLvlStr7 > 280,	metaData.FLIBconf_currLvlStr7 = 500;  end
                    metaData.FLIBconf_currLvlStr8 = bitand(temp, 2^30 + 2^31, 'uint32')/(2^30)*140;
                    if metaData.FLIBconf_currLvlStr8 > 280,	metaData.FLIBconf_currLvlStr8 = 500;  end
                    
                    metaDataBin.FLIBconf_flashCtrlStr1 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_flashCtrlStr2 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_flashCtrlStr3 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_flashCtrlStr4 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_flashCtrlStr5 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_flashCtrlStr6 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_flashCtrlStr7 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_flashCtrlStr8 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_xsubCtrlStr1 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_xsubCtrlStr2 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_xsubCtrlStr3 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_xsubCtrlStr4 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_xsubCtrlStr5 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_xsubCtrlStr6 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_xsubCtrlStr7 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_xsubCtrlStr8 = dec2hex(temp, 8);
                    metaDataBin.FLIBconf_currLvlStr1 = dec2hex(temp, 8);
                    
                case hex2dec('0302')    %   FLI_D1D2
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIconf_D1flashT = dec2hex(temp, 4);
                    metaData.FLIconf_D1flashT = double(temp)/(2*4.608e6);
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIconf_D2flashT = dec2hex(temp, 4);
                    metaData.FLIconf_D2flashT = double(temp)/(2*4.608e6);
                case hex2dec('0303')    %   FLI_D3D4
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIconf_D3flashT = dec2hex(temp, 4);
                    metaData.FLIconf_D3flashT = double(temp)/(2*4.608e6);
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIconf_D4flashT = dec2hex(temp, 4);
                    metaData.FLIconf_D4flashT = double(temp)/(2*4.608e6);
                case hex2dec('0304')    %   FLI_D5D6
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIconf_D5flashT = dec2hex(temp, 4);
                    metaData.FLIconf_D5flashT = double(temp)/(2*4.608e6);
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIconf_D6flashT = dec2hex(temp, 4);
                    metaData.FLIconf_D6flashT = double(temp)/(2*4.608e6);
                case hex2dec('0305')    %   FLI_D7D8
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIconf_D7flashT = dec2hex(temp, 4);
                    metaData.FLIconf_D7flashT = double(temp)/(2*4.608e6);
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIconf_D8flashT = dec2hex(temp, 4);
                    metaData.FLIconf_D8flashT = double(temp)/(2*4.608e6);
                case hex2dec('0400')    %   FLIHK1_A
                    temp = fread(fid,1,'*uint8');
                    metaDataBin.FLIA_HK_sync = dec2hex(temp, 2);
                    metaData.FLIA_HK_sync = temp;
                    temp = fread(fid,1,'*uint8');
                    metaDataBin.FLIA_HK_fieldCnt = dec2hex(temp, 2);
                    metaData.FLIA_HK_fieldCnt = temp;
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_flashTMax = dec2hex(temp, 4);
                    metaData.FLIA_HK_flashTMax = double(bitand(temp, 16383, 'uint16'))/(4.608e6);
                case hex2dec('0401')    %   FLIHK1_B
                    temp = fread(fid,1,'*uint8');
                    metaData.FLIB_HK_sync = temp;
                    metaDataBin.FLIB_HK_sync = dec2hex(temp, 2);
                    temp = fread(fid,1,'*uint8');
                    metaData.FLIB_HK_fieldCnt = temp;
                    metaDataBin.FLIB_HK_fieldCnt = dec2hex(temp, 2);
                    temp = fread(fid,1,'*uint16');
                    metaData.FLIB_HK_flashTMax = double(bitand(temp, 16383, 'uint16'))/(4.608e6);
                    metaDataBin.FLIB_HK_flashTMax = dec2hex(temp, 4);
                case hex2dec('0402')    %   FLIHK2_A
                    temp = fread(fid,1,'*uint16');
                    metaData.FLIA_HK_preF_ADCoff = temp;
                    metaDataBin.FLIA_HK_preF_ADCoff = dec2hex(temp, 4);
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_preF_temp = dec2hex(temp, 4);
                    metaData.FLIA_HK_preF_temp = double(temp)*0.0389-270;
                case hex2dec('0403')    %   FLIHK2_B
                    temp = fread(fid,1,'*uint16');
                    metaData.FLIB_HK_preF_ADCoff = temp;
                    metaDataBin.FLIB_HK_preF_ADCoff = dec2hex(temp, 4);
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIB_HK_preF_temp = dec2hex(temp, 4);
                    metaData.FLIB_HK_preF_temp = double(temp)*0.0389-270;
                case hex2dec('0404')    %   FLIHK3_A
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_preF_volBank1 = dec2hex(temp, 4);
                    metaData.FLIA_HK_preF_volBank1 = double(temp)*1.19e-3;
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_preF_volBank2 = dec2hex(temp, 4);
                    metaData.FLIA_HK_preF_volBank2 = double(temp)*1.19e-3;
                case hex2dec('0405')    %   FLIHK3_B
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIB_HK_preF_volBank1 = dec2hex(temp, 4);
                    metaData.FLIB_HK_preF_volBank1 = double(temp)*1.19e-3;
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIB_HK_preF_volBank2 = dec2hex(temp, 4);
                    metaData.FLIB_HK_preF_volBank2 = double(temp)*1.19e-3;
                case hex2dec('0406')    %   FLIHK4_A
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_preF_volBank3 = dec2hex(temp, 4);
                    metaData.FLIA_HK_preF_volBank3 = double(temp)*1.19e-3;
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_preF_volBank4 = dec2hex(temp, 4);
                    metaData.FLIA_HK_preF_volBank4 = double(temp)*1.19e-3;
                case hex2dec('0407')    %   FLIHK4_B
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIB_HK_preF_volBank3 = dec2hex(temp, 4);
                    metaData.FLIB_HK_preF_volBank3 = double(temp)*1.19e-3;
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIB_HK_preF_volBank4 = dec2hex(temp, 4);
                    metaData.FLIB_HK_preF_volBank4 = double(temp)*1.19e-3;
                case hex2dec('0408')    %   FLIHK5_A
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_postF_ADCoff = dec2hex(temp, 4);
                    metaData.FLIA_HK_postF_ADCoff = temp;
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_postF_temp = dec2hex(temp, 4);
                    metaData.FLIA_HK_postF_temp = double(temp)*0.0389-270;
                case hex2dec('0409')    %   FLIHK5_B
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIB_HK_postF_ADCoff = dec2hex(temp, 4);
                    metaData.FLIB_HK_postF_ADCoff = temp;
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIB_HK_postF_temp = dec2hex(temp, 4);
                    metaData.FLIB_HK_postF_temp = double(temp)*0.0389-270;
                case hex2dec('040A')    %   FLIHK6_A
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_postF_volBank1 = dec2hex(temp, 4);
                    metaData.FLIA_HK_postF_volBank1 = double(temp)*1.19e-3;
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_postF_volBank2 = dec2hex(temp, 4);
                    metaData.FLIA_HK_postF_volBank2 = double(temp)*1.19e-3;
                case hex2dec('040B')    %   FLIHK6_B
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIB_HK_postF_volBank1 = dec2hex(temp, 4);
                    metaData.FLIB_HK_postF_volBank1 = double(temp)*1.19e-3;
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIB_HK_postF_volBank2 = dec2hex(temp, 4);
                    metaData.FLIB_HK_postF_volBank2 = double(temp)*1.19e-3;
                case hex2dec('040C')    %   FLIHK7_A
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_postF_volBank3 = dec2hex(temp, 4);
                    metaData.FLIA_HK_postF_volBank3 = double(temp)*1.19e-3;
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIA_HK_postF_volBank4 = dec2hex(temp, 4);
                    metaData.FLIA_HK_postF_volBank4 = double(temp)*1.19e-3;
                case hex2dec('040D')    %   FLIHK7_B
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIB_HK_postF_volBank3 = dec2hex(temp, 4);
                    metaData.FLIB_HK_postF_volBank3 = double(temp)*1.19e-3;
                    temp = fread(fid,1,'*uint16');
                    metaDataBin.FLIB_HK_postF_volBank4 = dec2hex(temp, 4);
                    metaData.FLIB_HK_postF_volBank4 = double(temp)*1.19e-3;
                case hex2dec('040E')    %   FLIHK8_A
                    temp = fread(fid,1,'*uint32');
                    metaDataBin.FLIA_curr = dec2hex(temp, 8);
                    metaData.FLIA_curr = temp;
                case hex2dec('040F')    %   FLIHK8_B
                    temp = fread(fid,1,'*uint32');
                    metaDataBin.FLIB_curr = dec2hex(temp, 8);
                    metaData.FLIB_curr = temp;
                otherwise
                    fread(fid,1,'*uint32');
            end
        end
    end
end

%% --- Export SLI points to CSV (if available) ------------------------------

if strcmp(imageFormat, 'sli') && isfield(im, 'sliData')

    % Convert normalized SLI center coords to pixel coordinates
    cen = double(im.sliData.cen); % 1e3-scaled normalized centroids for SLI points
    u = cen(:,1) / 1000 * header.W;   % 0..1 → 0..W-1  (horizontal, OK already)
    v = cen(:,2) / 1000 * header.H;   % 0..1 → 0..H-1  (vertical, this was wrong)
    
    % 3D SLI point data
    X = im.sliData.Q(:,1);
    Y = im.sliData.Q(:,2);
    Z = im.sliData.Q(:,3);

    M = [u, v, X, Y, Z];

    % CSV filename matches input file, but with _SLI_points.csv appended
    [p,name,~] = fileparts(file);
    csvname = fullfile(p, [name '_SLI_points.csv']);

    % Write header
    fidCSV = fopen(csvname, 'w');
    fprintf(fidCSV, 'u,v,X,Y,Z\n');
    fclose(fidCSV);

    % Append data
    dlmwrite(csvname, M, '-append', 'delimiter', ',', 'precision', 9);

    fprintf('Wrote SLI CSV: %s\n', csvname);
end


fclose(fid);