% Process all unc files in pwd and generate pngs with LED/channel labels
files = dir('*.unc'); % Get all .unc files in the current directory

prefix = "Obstacle_040mm_"; % Change between Calibration, NoObstacle and Obstacle for each dataset OBJECTTYPE

for k = 1:numel(files)
    fileName = files(k).name;

    fprintf('\n=== Processing: %s ===\n', fileName);

    % Run readAscImage.m
    try
        [header, im, metaData, metaDataBin] = readAscImage(fileName); % preferred
    catch

    end

    % Ensure that our metaData is not none
    if isstruct(metaData)
        fprintf('  metaData fields: %d\n', numel(fieldnames(metaData)));
    else
        fprintf('  metaData type: %s\n', class(metaData));
    end

    % Convert to 8-bit PNG
    I8 = uint8(255 * mat2gray(im));

    %% Extract active D channels from metaData for a given .unc
    activeD = [];

    for d = 1:8
        f = sprintf('FLIconf_D%dflashT', d);

        
        if isstruct(metaData) && isfield(metaData, f) && isnumeric(metaData.(f))
            if metaData.(f) > 0
                activeD(end+1) = d;
            end

        % Fallback to hex metaDataBin if present
        elseif isstruct(metaDataBin) && isfield(metaDataBin, f)
            v = metaDataBin.(f); % e.g. 'FFFF' or '0000'
            if isstring(v); v = char(v); end
            if ischar(v) && ~isempty(v)
                % hex2dec expects valid hex; guard against weird values
                try
                    if hex2dec(v) > 0
                        activeD(end+1) = d; %#ok<SAGROW>
                    end
                catch
                    % ignore if not valid hex
                end
            end
        end
    end

    % Build label string
    if isempty(activeD)
        fprintf('  Active D channels : none\n');
        ledLabel = "Dnone";
    else
        fprintf('  Active D channels : D%s\n', strjoin(string(activeD), ' D'));
        if numel(activeD) == 1
            ledLabel = "D" + activeD;
        elseif all(diff(activeD) == 1)
            ledLabel = "D" + activeD(1) + "-D" + activeD(end);
        else
            ledLabel = "D" + strjoin(string(activeD), "_D");
        end
    end

    %% Extract active strings from A & B banks
    % This relies on metaData fields not being empty
    activeStrA = [];
    activeStrB = [];

    for s = 1:8
        fA_en = sprintf('FLIAconf_xsubCtrlStr%d', s);
        fA_I  = sprintf('FLIAconf_currLvlStr%d', s);
        fB_en = sprintf('FLIBconf_xsubCtrlStr%d', s);
        fB_I  = sprintf('FLIBconf_currLvlStr%d', s);

        if isstruct(metaData) && isfield(metaData, fA_en) && isfield(metaData, fA_I)
            if metaData.(fA_en) == 1 && metaData.(fA_I) > 0
                activeStrA(end+1) = s;
            end
        end

        if isstruct(metaData) && isfield(metaData, fB_en) && isfield(metaData, fB_I)
            if metaData.(fB_en) == 1 && metaData.(fB_I) > 0
                activeStrB(end+1) = s;
            end
        end
    end

    % Print strings
    if isempty(activeStrA)
        fprintf('  Active Str (A)    : none\n');
    else
        fprintf('  Active Str (A)    : %s\n', strjoin(string(activeStrA), ' '));
    end

    if isempty(activeStrB)
        fprintf('  Active Str (B)    : none\n');
    else
        fprintf('  Active Str (B)    : %s\n', strjoin(string(activeStrB), ' '));
    end

    %% Infer illumination mode
    if numel(activeD) == 1
        illumMode = "single-LED (directional)";
    elseif numel(activeD) > 1
        illumMode = "multi-LED (composite)";
    else
        illumMode = "unknown / dark";
    end

    % Refine using odd/even flags if present (numeric metaData)
    if isstruct(metaData) && isfield(metaData, 'SLIFLIctrl_fliAstrOdd') && isfield(metaData, 'SLIFLIctrl_fliAstrEven')
        if metaData.SLIFLIctrl_fliAstrEven && ~metaData.SLIFLIctrl_fliAstrOdd
            illumMode = illumMode + ", even-strings";
        elseif metaData.SLIFLIctrl_fliAstrOdd && ~metaData.SLIFLIctrl_fliAstrEven
            illumMode = illumMode + ", odd-strings";
        end
    end

    fprintf('  Illumination mode : %s\n', illumMode);

    %% Write PNG
    outName = sprintf('%s%04d_%s.png', prefix, k, ledLabel);
    imwrite(I8, outName);
end
