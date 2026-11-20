%% residual_metrics.m
% Residual swing angle after the command stop, for all 100 runs in
% data/experiments (rope lengths 0.50 to 1.30 m, methods Pulse, ZV, ZVD and AIS).
%
% For each raw.csv the residual window runs from 0.20 s to 10.20 s after the
% command stop. The angle is resampled at 0.01 s and demeaned.
%
% Metrics:
%   1) peak-to-peak residual swing angle [deg]
%   2) RMS residual swing angle [deg]
%   3) residual energy integral(theta^2 dt) [deg^2*s]
%
% Each metric is also divided by the median Pulse value at the same rope length.
%
% Outputs are written to experiments/out/residual_metrics.

clear; clc;

%% ---------------- USER SETTINGS ----------------

REPO_DIR = fileparts(fileparts(mfilename("fullpath")));
BASE_DIR = fullfile(REPO_DIR, "data", "experiments");

OUT_DIR = fullfile(REPO_DIR, "experiments", "out", "residual_metrics");
if ~exist(OUT_DIR, "dir"); mkdir(OUT_DIR); end

% Rope lengths [m].
LENGTH_ORDER = [0.50 0.70 0.90 1.10 1.30];

METHOD_ORDER = ["Pulse", "ZV", "ZVD", "AIS"];

% Residual gate after the command stop [s].
GATE_LIST_S = 0.20;

% Residual duration after gate.
RESIDUAL_DURATION_S = 10.00;
DT_RESAMPLE = 0.01;

%% ---------------- DISCOVER raw.csv FILES ----------------

rawFiles = strings(0,1);

for iL = 1:numel(LENGTH_ORDER)
    L = LENGTH_ORDER(iL);
    Lfolder = sprintf("L%03d", round(100*L));

    thisRoot = fullfile(BASE_DIR, Lfolder);
    d = dir(fullfile(thisRoot, "**", "raw.csv"));

    for k = 1:numel(d)
        rawFiles(end+1,1) = string(fullfile(d(k).folder, d(k).name)); %#ok<SAGROW>
    end
end

rawFiles = unique(rawFiles, "stable");

fprintf("Found %d raw.csv files.\n", numel(rawFiles));
fprintf("Output folder:\n%s\n\n", OUT_DIR);

if isempty(rawFiles)
    error("No raw.csv files found. Check BASE_DIR.");
end

%% ---------------- PROCESS FILES ----------------

results = table();
failed = table();

for i = 1:numel(rawFiles)

    rawFile = rawFiles(i);
    fprintf("\n[%03d/%03d] %s\n", i, numel(rawFiles), rawFile);

    try
        info = parseRunInfoFromPath(rawFile);

        if ~ismember(info.method, METHOD_ORDER)
            warning("Unknown method. Skipping.");
            failed = appendFail(failed, rawFile, "unknown_method");
            continue;
        end

        if ~ismember(round(info.L_m,2), LENGTH_ORDER)
            warning("Unknown length. Skipping.");
            failed = appendFail(failed, rawFile, "unknown_length");
            continue;
        end

        data = loadRawFlexible(rawFile, info.L_m);

        for iGate = 1:numel(GATE_LIST_S)

            gateS = GATE_LIST_S(iGate);

            metrics = computeRawResidualMetrics( ...
                data, ...
                gateS, ...
                RESIDUAL_DURATION_S, ...
                DT_RESAMPLE);

            row = metricsToRow(metrics, rawFile, info, gateS);
            results = [results; row]; %#ok<AGROW>

            fprintf("  gate=%.2fs | L=%.2f | %-13s | rep%02d | p2p=%.4f deg | RMS=%.4f deg | E=%.4g deg^2*s | duration=%.2fs\n", ...
                gateS, info.L_m, info.method, info.rep, ...
                metrics.p2p_swing_deg, ...
                metrics.rms_swing_deg, ...
                metrics.residual_energy_deg2_s, ...
                metrics.residual_duration_actual_s);
        end

    catch ME
        warning("Failed run:\n%s\nReason: %s", rawFile, ME.message);
        failed = appendFail(failed, rawFile, string(ME.message));
    end
end

%% ---------------- SAVE RAW CSVs ----------------

outCsv = fullfile(OUT_DIR, "raw_10s_residual_angle_energy_results_gates.csv");
writetable(results, outCsv);

outFail = fullfile(OUT_DIR, "failed_runs.csv");
writetable(failed, outFail);

fprintf("\nSaved raw results:\n%s\n", outCsv);

if ~isempty(failed)
    fprintf("Saved failed runs:\n%s\n", outFail);
end

%% ---------------- RAW SUMMARY CSV ----------------

if ~isempty(results)

    summaryRaw = groupsummary(results, ["gate_after_stop_s", "length_m", "method"], ["mean", "std", "median"], ...
        ["p2p_swing_deg", ...
         "rms_swing_deg", ...
         "max_abs_swing_deg", ...
         "residual_energy_deg2_s"]);

    outSummaryRaw = fullfile(OUT_DIR, "raw_10s_summary_by_gate_length_method.csv");
    writetable(summaryRaw, outSummaryRaw);

    fprintf("\nRaw summary by gate/length/method:\n");
    disp(summaryRaw);
end

%% ---------------- NORMALIZE BY PULSE MEDIAN ----------------

if ~isempty(results)

    resultsNorm = normalizeByPulseMedian(results);

    outNormCsv = fullfile(OUT_DIR, "raw_10s_residual_angle_energy_results_gates_NORMALIZED_BY_PULSE.csv");
    writetable(resultsNorm, outNormCsv);

    fprintf("\nSaved pulse-normalized results:\n%s\n", outNormCsv);

    summaryNorm = groupsummary(resultsNorm, ["gate_after_stop_s", "length_m", "method"], ["mean", "std", "median"], ...
        ["p2p_swing_norm_pulse", ...
         "rms_swing_norm_pulse", ...
         "residual_energy_norm_pulse"]);

    outSummaryNorm = fullfile(OUT_DIR, "raw_10s_summary_NORMALIZED_BY_PULSE.csv");
    writetable(summaryNorm, outSummaryNorm);

    fprintf("\nPulse-normalized summary:\n");
    disp(summaryNorm);
else
    resultsNorm = table();
end

fprintf("\nDONE.\nOutput folder:\n%s\n", OUT_DIR);


%% ========================================================================
% Local functions
%% ========================================================================

function info = parseRunInfoFromPath(rawFile)

    % Parse the length folder and run folder only, so the location of the
    % repository does not matter.
    [runFolder, ~, ~] = fileparts(rawFile);
    [lengthFolder, runFolderName] = fileparts(runFolder);
    [~, lengthFolderName] = fileparts(lengthFolder);
    s = lower("/" + string(lengthFolderName) + "/" + string(runFolderName) + "/raw.csv");

    info.L_m = NaN;
    info.method = "Unknown";
    info.rep = NaN;

    % The rope length comes from physical_L050 and so on in the run name.
    tokPhys = regexp(s, "physical_l(\d{3})", "tokens", "once");
    tokFolder = regexp(s, "[\\/]+l(\d{3})[\\/]+", "tokens", "once");
    tokAny = regexp(s, "l(\d{3})", "tokens", "once");

    if ~isempty(tokPhys)
        info.L_m = str2double(tokPhys{1}) / 100;
    elseif ~isempty(tokFolder)
        info.L_m = str2double(tokFolder{1}) / 100;
    elseif ~isempty(tokAny)
        info.L_m = str2double(tokAny{1}) / 100;
    end

    % Order matters: "nonrobust" contains "robust".
    if contains(s, "pulse")
        info.method = "Pulse";
    elseif contains(s, "nonrobust") || contains(s, "_zv_")
        info.method = "ZV";
    elseif contains(s, "robust")
        info.method = "ZVD";
    elseif contains(s, "zerozeta") || contains(s, "zero_zeta") || contains(s, "is2") || contains(s, "isa")
        info.method = "AIS";
    else
        info.method = "Unknown";
    end

    tokRep = regexp(s, "rep(\d+)", "tokens", "once");
    if ~isempty(tokRep)
        info.rep = str2double(tokRep{1});
    end

    [folder, ~, ~] = fileparts(rawFile);
    [~, runName] = fileparts(folder);
    info.run_name = string(runName);
end

function data = loadRawFlexible(rawFile, L_m)

    opts = detectImportOptions(rawFile, "FileType", "text");
    opts.VariableNamingRule = "preserve";
    T = readtable(rawFile, opts);

    names = string(T.Properties.VariableNames);

    data.t = getFirstAvailableNumeric(T, names, [
        "move_time_sec"
        "t_motion_sec"
        "time_sec"
        "wall_time_sec"
        "stamp_sec"
        "t"
    ]);

    data.cmd_vx = getFirstAvailableNumericOptional(T, names, [
        "cmd_vx_mm_s"
        "vx_cmd_mm_s"
        "cmd_vel_x_mm_s"
        "cmd_vx"
    ]);

    data.cmd_vy = getFirstAvailableNumericOptional(T, names, [
        "cmd_vy_mm_s"
        "vy_cmd_mm_s"
        "cmd_vel_y_mm_s"
        "cmd_vy"
    ]);

    data.swing_mm = getFirstAvailableNumericOptional(T, names, [
        "swing_mm"
        "payload_rel_mm"
        "payload_x_rel_mm"
        "rel_payload_mm"
        "x_rel_mm"
    ]);

    [data.angle_deg, data.angle_source] = getAngleSignalDeg(T, names, L_m, data.swing_mm);

    n = numel(data.t);

    data.cmd_vx = resizeToN(data.cmd_vx, n);
    data.cmd_vy = resizeToN(data.cmd_vy, n);
    data.swing_mm = resizeToN(data.swing_mm, n);
    data.angle_deg = resizeToN(data.angle_deg, n);

    if ~any(isfinite(data.cmd_vx)); data.cmd_vx = zeros(n,1); end
    if ~any(isfinite(data.cmd_vy)); data.cmd_vy = zeros(n,1); end

    valid = isfinite(data.t) & isfinite(data.angle_deg);

    data.t = data.t(valid);
    data.cmd_vx = data.cmd_vx(valid);
    data.cmd_vy = data.cmd_vy(valid);
    data.swing_mm = data.swing_mm(valid);
    data.angle_deg = data.angle_deg(valid);

    data.t = data.t - data.t(1);

    [data.t, ia] = unique(data.t, "stable");
    data.cmd_vx = data.cmd_vx(ia);
    data.cmd_vy = data.cmd_vy(ia);
    data.swing_mm = data.swing_mm(ia);
    data.angle_deg = data.angle_deg(ia);
end

function metrics = computeRawResidualMetrics(data, guardS, durationS, dt)

    t = data.t(:);
    yAll = data.angle_deg(:);

    cmdMag = hypot(data.cmd_vx(:), data.cmd_vy(:));
    cmdMag(~isfinite(cmdMag)) = 0;

    tf = detectTf(t, cmdMag);

    tStart = tf + guardS;
    tEndRequested = tf + guardS + durationS;
    tEnd = min(tEndRequested, t(end));

    tqAbs = (tStart:dt:tEnd).';

    if numel(tqAbs) < 50
        error("Not enough residual samples.");
    end

    thetaRaw = interp1(t, yAll, tqAbs, "linear", "extrap");

    % Demean so the encoder and camera offset does not dominate the energy.
    theta = thetaRaw - mean_omitnan(thetaRaw);

    tr = tqAbs - tqAbs(1);

    metrics.tf_s = tf;
    metrics.residual_start_abs_s = tStart;
    metrics.residual_end_abs_s = tEnd;
    metrics.residual_duration_actual_s = tEnd - tStart;
    metrics.requested_duration_s = durationS;
    metrics.n_samples = numel(theta);

    metrics.p2p_swing_deg = max(theta) - min(theta);
    metrics.rms_swing_deg = rms_omitnan(theta);
    metrics.max_abs_swing_deg = max(abs(theta));
    metrics.residual_energy_deg2_s = trapz(tr, theta.^2);

    metrics.mean_raw_angle_deg = mean_omitnan(thetaRaw);
    metrics.angle_source = string(data.angle_source);
end

function row = metricsToRow(m, rawFile, info, gateS)

    row = table();

    row.raw_file = string(rawFile);
    row.run_name = string(info.run_name);
    row.length_m = info.L_m;
    row.length_label = "L" + string(round(100*info.L_m));
    row.method = string(info.method);
    row.rep = info.rep;

    row.gate_after_stop_s = gateS;

    row.tf_s = m.tf_s;
    row.residual_start_abs_s = m.residual_start_abs_s;
    row.residual_end_abs_s = m.residual_end_abs_s;
    row.residual_duration_actual_s = m.residual_duration_actual_s;
    row.requested_duration_s = m.requested_duration_s;
    row.n_samples = m.n_samples;

    row.p2p_swing_deg = m.p2p_swing_deg;
    row.rms_swing_deg = m.rms_swing_deg;
    row.max_abs_swing_deg = m.max_abs_swing_deg;
    row.residual_energy_deg2_s = m.residual_energy_deg2_s;

    row.mean_raw_angle_deg = m.mean_raw_angle_deg;
    row.angle_source = m.angle_source;
end

function resultsNorm = normalizeByPulseMedian(results)

    resultsNorm = results;

    resultsNorm.p2p_swing_norm_pulse = nan(height(results), 1);
    resultsNorm.rms_swing_norm_pulse = nan(height(results), 1);
    resultsNorm.residual_energy_norm_pulse = nan(height(results), 1);

    gates = unique(results.gate_after_stop_s);
    lengths = unique(results.length_m);

    for iG = 1:numel(gates)

        gateS = gates(iG);

        for iL = 1:numel(lengths)

            L = lengths(iL);

            idxBase = abs(results.gate_after_stop_s - gateS) < 1e-9 & ...
                      abs(results.length_m - L) < 1e-9 & ...
                      results.method == "Pulse";

            if ~any(idxBase)
                warning("No Pulse baseline found for gate=%.2f, L=%.2f. Normalized values set to NaN.", gateS, L);
                continue;
            end

            baseP2P = median(results.p2p_swing_deg(idxBase), "omitnan");
            baseRMS = median(results.rms_swing_deg(idxBase), "omitnan");
            baseEnergy = median(results.residual_energy_deg2_s(idxBase), "omitnan");

            idxAll = abs(results.gate_after_stop_s - gateS) < 1e-9 & ...
                     abs(results.length_m - L) < 1e-9;

            resultsNorm.p2p_swing_norm_pulse(idxAll) = results.p2p_swing_deg(idxAll) ./ max(baseP2P, eps);
            resultsNorm.rms_swing_norm_pulse(idxAll) = results.rms_swing_deg(idxAll) ./ max(baseRMS, eps);
            resultsNorm.residual_energy_norm_pulse(idxAll) = results.residual_energy_deg2_s(idxAll) ./ max(baseEnergy, eps);
        end
    end
end

function failed = appendFail(failed, rawFile, reason)

    row = table();
    row.raw_file = string(rawFile);
    row.reason = string(reason);

    failed = [failed; row];
end

function tf = detectTf(t, cmdMag)

    maxCmd = max(cmdMag);

    if ~isfinite(maxCmd) || maxCmd <= 1e-9
        error("No command motion found.");
    end

    thresh = max(1e-6, 0.05 * maxCmd);

    nonzeroSeen = false;

    for i = 1:numel(cmdMag)
        if cmdMag(i) > thresh
            nonzeroSeen = true;
        end

        if nonzeroSeen && cmdMag(i) <= thresh
            tf = t(i);
            return;
        end
    end

    error("Could not detect stop time.");
end

function [angle_deg, source] = getAngleSignalDeg(T, names, L_m, swing_mm)

    if any(names == "swing_pitch_deg")
        angle_deg = forceNumeric(T.("swing_pitch_deg"));
        source = "swing_pitch_deg";
        return;
    end

    if any(names == "swing_axis_angle_deg")
        angle_deg = forceNumeric(T.("swing_axis_angle_deg"));
        source = "swing_axis_angle_deg";
        return;
    end

    if any(names == "enc_pitch_deg")
        angle_deg = forceNumeric(T.("enc_pitch_deg"));
        source = "enc_pitch_deg";
        return;
    end

    if any(names == "payload_pitch_deg")
        angle_deg = forceNumeric(T.("payload_pitch_deg"));
        source = "payload_pitch_deg";
        return;
    end

    if any(names == "enc_roll_deg")
        angle_deg = forceNumeric(T.("enc_roll_deg"));
        source = "enc_roll_deg";
        return;
    end

    if any(isfinite(swing_mm)) && isfinite(L_m)
        angle_rad = (swing_mm ./ 1000) ./ L_m;
        angle_deg = rad2deg(angle_rad);
        source = "swing_mm_converted_to_deg";
        return;
    end

    error("No usable angle signal found.");
end

function x = getFirstAvailableNumeric(T, names, candidates)

    x = getFirstAvailableNumericOptional(T, names, candidates);

    if ~any(isfinite(x))
        error("None of these required columns found: %s", strjoin(candidates, ", "));
    end
end

function x = getFirstAvailableNumericOptional(T, names, candidates)

    candidates = string(candidates);
    x = nan(height(T), 1);

    for i = 1:numel(candidates)

        c = candidates(i);

        if any(names == c)
            x = forceNumeric(T.(c));
            return;
        end
    end
end

function x = forceNumeric(x)

    if isnumeric(x) || islogical(x)
        x = double(x);
        x = x(:);
        return;
    end

    if iscell(x)
        x = string(x);
    end

    if isstring(x) || ischar(x) || iscategorical(x)
        x = str2double(string(x));
        x = x(:);
        return;
    end

    error("Invalid data type.");
end

function x = resizeToN(x, n)

    x = x(:);

    if numel(x) == n
        return;
    end

    if isempty(x)
        x = nan(n,1);
        return;
    end

    if numel(x) > n
        x = x(1:n);
    else
        x(end+1:n,1) = NaN;
    end
end

function r = rms_omitnan(x)

    x = x(:);
    x = x(isfinite(x));

    if isempty(x)
        r = NaN;
    else
        r = sqrt(mean(x.^2));
    end
end

function m = mean_omitnan(x)

    x = x(:);
    x = x(isfinite(x));

    if isempty(x)
        m = NaN;
    else
        m = mean(x);
    end
end