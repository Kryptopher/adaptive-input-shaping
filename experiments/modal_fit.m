%% modal_fit.m
% Two-mode fit of the residual swing angle after the command stop, for all 100
% runs in data/experiments (rope lengths 0.50 to 1.30 m, methods Pulse, ZV,
% ZVD and AIS).
%
% For each raw.csv:
%   1) Detect the command stop time tf
%   2) Take the residual window from tf + 0.20 s to tf + 10.20 s
%   3) Run 10 initialized two-mode fits
%   4) Weight early samples more, because mode 2 is mostly an early transient
%   5) Use a cost with a dead band of half an encoder tick (tick 0.09 deg)
%   6) Select the best fit by
%          selection_score = quantized objective
%                          + 0.10 * weighted raw RMSE
%                          + 0.05 * RMSE over the first 0.5 s
%                          + 0.02 * RMSE over the first 2.0 s
%   7) Save the modal metrics and the decomposition of the selected fit
%
% Outputs are written to experiments/out/modal_fit.

clear; clc;

%% ---------------- USER SETTINGS ----------------

REPO_DIR = fileparts(fileparts(mfilename("fullpath")));
BASE_DIR = fullfile(REPO_DIR, "data", "experiments");

OUT_DIR = fullfile(REPO_DIR, "experiments", "out", "modal_fit");
if ~exist(OUT_DIR, "dir"); mkdir(OUT_DIR); end

FIT_DIR = fullfile(OUT_DIR, "best_fit_decompositions");
if ~exist(FIT_DIR, "dir"); mkdir(FIT_DIR); end

LENGTH_ORDER = [0.50 0.70 0.90 1.10 1.30];
METHOD_ORDER = ["Pulse", "ZV", "ZVD", "AIS"];
GATE_LIST_S = 0.20;

RESIDUAL_DURATION_S = 10.00;
DT_RESAMPLE = 0.01;
N_INIT = 10;
SMOOTH_S = 0.03;
F2_MIN_BASE = 1.20;
F2_MAX = 20.00;

%% ---------------- EARLY TRANSIENT WEIGHTING ----------------
USE_EARLY_WEIGHTED_FIT = true;
VERY_EARLY_WINDOW_S = 0.50;
EARLY_WINDOW_S = 2.00;
VERY_EARLY_WEIGHT = 8.0;
EARLY_WEIGHT = 4.0;
LATE_WEIGHT = 1.0;

%% ---------------- QUANTIZATION-AWARE FITTING ----------------
USE_QUANTIZATION_AWARE_COST = true;
QUANTIZATION_DEG = 0.09;
QUANTIZATION_DEADBAND_DEG = 0.045;   % half a tick, so fewer fits reach a zero objective

%% ---------------- QUANTIZATION-AWARE REPORTING ----------------
% These thresholds affect only the reported energies, not the fit. If mode 2 is
% at the scale of one encoder tick, the fit is kept but the reported mode-2
% energy is zero, because such a mode is quantization, interpolation or a fit
% to the stair steps. If the whole residual is at that scale, all reported
% energies are zero.
MODE2_QUANT_MAX_AMP_DEG = 1.25 * QUANTIZATION_DEG;
MODE2_QUANT_MAX_P2P_DEG = 2.50 * QUANTIZATION_DEG;
MODE2_QUANT_MAX_RMS_DEG = 0.85 * QUANTIZATION_DEG;
WHOLE_RUN_QUANT_MAX_P2P_DEG = 2.00 * QUANTIZATION_DEG;
WHOLE_RUN_QUANT_MAX_RMS_DEG = 0.75 * QUANTIZATION_DEG;

%% ---------------- SELECTION SCORE WEIGHTS ----------------
SELECTION_WEIGHT_WEIGHTED_RMSE = 0.10;
SELECTION_WEIGHT_RMSE_0P5S = 0.05;
SELECTION_WEIGHT_RMSE_2S = 0.02;

%% ---------------- OPTIMIZER ----------------
optsFit = optimset("Display", "off", "MaxIter", 3000, "MaxFunEvals", 12000, "TolX", 1e-9, "TolFun", 1e-9);

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
fprintf("Output folder:\n%s\n", OUT_DIR);
fprintf("Best-fit decompositions:\n%s\n", FIT_DIR);
fprintf("\n");

if isempty(rawFiles)
    error("No raw.csv files found. Check BASE_DIR.");
end

%% ---------------- PROCESS FILES ----------------
results = table();
failed = table();
initLog = table();

for i = 1:numel(rawFiles)
    rawFile = rawFiles(i);
    fprintf("\n============================================================\n");
    fprintf("[%03d/%03d] %s\n", i, numel(rawFiles), rawFile);
    fprintf("============================================================\n");

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
            bestMetrics = [];
            bestScore = inf;
            fprintf("\n  Gate %.2f s: running %d initialized fits...\n", gateS, N_INIT);

            for init = 1:N_INIT
                rng(1000000 + 10000*i + 100*iGate + init);
                try
                    m = modalFitOneRun(data, info.L_m, gateS, RESIDUAL_DURATION_S, DT_RESAMPLE, SMOOTH_S, ...
                        F2_MIN_BASE, F2_MAX, optsFit, init, ...
                        USE_EARLY_WEIGHTED_FIT, VERY_EARLY_WINDOW_S, EARLY_WINDOW_S, ...
                        VERY_EARLY_WEIGHT, EARLY_WEIGHT, LATE_WEIGHT, ...
                        USE_QUANTIZATION_AWARE_COST, QUANTIZATION_DEG, QUANTIZATION_DEADBAND_DEG, ...
                        MODE2_QUANT_MAX_AMP_DEG, MODE2_QUANT_MAX_P2P_DEG, MODE2_QUANT_MAX_RMS_DEG, ...
                        WHOLE_RUN_QUANT_MAX_P2P_DEG, WHOLE_RUN_QUANT_MAX_RMS_DEG);

                    m.init_number = init;
                    m.selection_score = m.fit_objective_cost ...
                        + SELECTION_WEIGHT_WEIGHTED_RMSE * m.weighted_raw_rmse_deg ...
                        + SELECTION_WEIGHT_RMSE_0P5S * m.two_mode_rmse_first_0p5s_deg ...
                        + SELECTION_WEIGHT_RMSE_2S * m.two_mode_rmse_first_2s_deg;

                    initRow = initializationLogRow(m, rawFile, info, gateS, init, true, "");
                    initLog = safeAppendTable(initLog, initRow);

                    fprintf("    init %02d | score=%.6f | obj=%.6f | wRMSE=%.6f | rawRMSE=%.6f | early0.5=%.6f | early2=%.6f | A1=%.4f f1=%.4f | A2=%.4f f2=%.4f\n", ...
                        init, m.selection_score, m.fit_objective_cost, m.weighted_raw_rmse_deg, ...
                        m.two_mode_rmse_deg, m.two_mode_rmse_first_0p5s_deg, m.two_mode_rmse_first_2s_deg, ...
                        abs(m.A1_deg), m.f1_damped_Hz, abs(m.A2_deg), m.f2_damped_Hz);

                    if m.selection_score < bestScore
                        bestScore = m.selection_score;
                        bestMetrics = m;
                        bestMetrics.accepted_init = init;
                        bestMetrics.selection_score = bestScore;
                    end

                catch MEfit
                    fprintf("    init %02d FAILED: %s\n", init, MEfit.message);
                    initRow = failedInitializationLogRow(rawFile, info, gateS, init, string(MEfit.message));
                    initLog = safeAppendTable(initLog, initRow);
                end
            end

            if isempty(bestMetrics)
                error("All %d modal initializations failed for gate %.2f.", N_INIT, gateS);
            end

            row = modalMetricsToRow(bestMetrics, rawFile, info, gateS);
            results = [results; row]; %#ok<AGROW>

            saveBestFitDecomposition(bestMetrics, rawFile, info, gateS, FIT_DIR);

            fprintf("  BEST gate=%.2fs | L=%.2f | %-13s | rep%02d | bestInit=%02d | score=%.6f | obj=%.6f | rawRMSE=%.6f | E1=%.4g | E2=%.4g | A1=%.4f | A2=%.4f | f1=%.3f | f2=%.3f\n", ...
                gateS, info.L_m, info.method, info.rep, bestMetrics.accepted_init, bestMetrics.selection_score, ...
                bestMetrics.fit_objective_cost, bestMetrics.two_mode_rmse_deg, bestMetrics.mode1_energy, ...
                bestMetrics.mode2_energy, abs(bestMetrics.A1_deg), abs(bestMetrics.A2_deg), ...
                bestMetrics.f1_damped_Hz, bestMetrics.f2_damped_Hz);
        end

    catch ME
        warning("Failed run:\n%s\nReason: %s", rawFile, ME.message);
        failed = appendFail(failed, rawFile, string(ME.message));
    end
end

%% ---------------- SAVE CSVs ----------------
outCsv = fullfile(OUT_DIR, "modal_10s_results_gates_bestof10.csv");
writetable(results, outCsv);
outFail = fullfile(OUT_DIR, "failed_runs.csv");
writetable(failed, outFail);
outInitLog = fullfile(OUT_DIR, "all_initializations_log.csv");
writetable(initLog, outInitLog);

fprintf("\nSaved modal results:\n%s\n", outCsv);
fprintf("Saved initialization log:\n%s\n", outInitLog);
if ~isempty(failed)
    fprintf("Saved failed runs:\n%s\n", outFail);
end

if ~isempty(results)
    quantRows = results(results.quantization_only_run | results.mode2_quant_artifact, :);
    outQuant = fullfile(OUT_DIR, "modal_10s_quantization_filtered_rows.csv");
    writetable(quantRows, outQuant);
    fprintf("Saved quantization-filtered rows:\n%s\n", outQuant);
end

%% ---------------- SUMMARY ----------------
if ~isempty(results)
    summaryRaw = groupsummary(results, ["gate_after_stop_s", "length_m", "method"], ["mean", "std", "median"], ...
        ["mode1_Eavg_reported_J_per_kg", "mode2_Eavg_reported_J_per_kg", "fit_total_Eavg_reported_J_per_kg", ...
         "mode1_Eavg_J_per_kg", "mode2_Eavg_J_per_kg", "fit_total_Eavg_J_per_kg", "measured_Eavg_J_per_kg", ...
         "A1_abs_deg", "A2_abs_deg", ...
         "f1_damped_Hz", "f2_damped_Hz", "selection_score", "fit_objective_cost", ...
         "weighted_raw_rmse_deg", "two_mode_rmse_deg", "two_mode_rmse_first_0p5s_deg", ...
         "two_mode_rmse_first_2s_deg", "residual_rms_over_quantization"]);
    outSummaryRaw = fullfile(OUT_DIR, "modal_10s_summary_by_gate_length_method.csv");
    writetable(summaryRaw, outSummaryRaw);
    fprintf("\nModal summary:\n");
    disp(summaryRaw);
end

fprintf("\nDONE.\nOutput folder:\n%s\n", OUT_DIR);
fprintf("Best-fit decomposition CSVs:\n%s\n", FIT_DIR);

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

    tokPhys = regexp(s, "physical_l(\d{3})", "tokens", "once");
    tokFolder = regexp(s, "[\\/]+l(\d{3})[\\/]+", "tokens", "once");
    tokAny = regexp(s, "l(\d{3})", "tokens", "once");

    % The rope length is the length folder (L050, L070, ...). Run names can
    % also name the design length, as in "robust_L090model".
    if ~isempty(tokFolder)
        info.L_m = str2double(tokFolder{1}) / 100;
    elseif ~isempty(tokPhys)
        info.L_m = str2double(tokPhys{1}) / 100;
    elseif ~isempty(tokAny)
        info.L_m = str2double(tokAny{1}) / 100;
    end

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

    data.t = getFirstAvailableNumeric(T, names, ["move_time_sec", "t_motion_sec", "time_sec", "wall_time_sec", "stamp_sec", "t"]);
    data.cmd_vx = getFirstAvailableNumericOptional(T, names, ["cmd_vx_mm_s", "vx_cmd_mm_s", "cmd_vel_x_mm_s", "cmd_vx"]);
    data.cmd_vy = getFirstAvailableNumericOptional(T, names, ["cmd_vy_mm_s", "vy_cmd_mm_s", "cmd_vel_y_mm_s", "cmd_vy"]);
    data.swing_mm = getFirstAvailableNumericOptional(T, names, ["swing_mm", "payload_rel_mm", "payload_x_rel_mm", "rel_payload_mm", "x_rel_mm"]);
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

function m = modalFitOneRun(data, L_m, gateS, durationS, DT, smoothS, F2_MIN_BASE, F2_MAX, optsFit, init, ...
    USE_EARLY_WEIGHTED_FIT, VERY_EARLY_WINDOW_S, EARLY_WINDOW_S, VERY_EARLY_WEIGHT, EARLY_WEIGHT, LATE_WEIGHT, ...
    USE_QUANTIZATION_AWARE_COST, QUANTIZATION_DEG, QUANTIZATION_DEADBAND_DEG, ...
    MODE2_QUANT_MAX_AMP_DEG, MODE2_QUANT_MAX_P2P_DEG, MODE2_QUANT_MAX_RMS_DEG, ...
    WHOLE_RUN_QUANT_MAX_P2P_DEG, WHOLE_RUN_QUANT_MAX_RMS_DEG)

    t = data.t(:);
    yAll = data.angle_deg(:);
    cmdMag = hypot(data.cmd_vx(:), data.cmd_vy(:));
    cmdMag(~isfinite(cmdMag)) = 0;
    tf = detectTf(t, cmdMag);

    tStart = tf + gateS;
    tEndRequested = tf + gateS + durationS;
    tEnd = min(tEndRequested, t(end));
    tqAbs = (tStart:DT:tEnd).';
    if numel(tqAbs) < 80
        error("Not enough residual samples.");
    end

    tr = tqAbs - tqAbs(1);
    yRaw = interp1(t, yAll, tqAbs, "linear", "extrap");
    y = yRaw - mean_omitnan(yRaw);

    if USE_EARLY_WEIGHTED_FIT
        wFit = LATE_WEIGHT * ones(size(tr));
        wFit(tr <= EARLY_WINDOW_S) = EARLY_WEIGHT;
        wFit(tr <= VERY_EARLY_WINDOW_S) = VERY_EARLY_WEIGHT;
        wFit = wFit ./ mean(wFit);
    else
        wFit = ones(size(tr));
    end

    if smoothS > 0
        winN = max(3, round(smoothS / DT));
        if mod(winN, 2) == 0; winN = winN + 1; end
        if exist("smoothdata", "file") || exist("smoothdata", "builtin")
            y = smoothdata(y, "movmean", winN);
        else
            y = movmean(y, winN, "omitnan");
        end
    end

    if rms_omitnan(y) < 1e-9
        error("Signal too small for modal fit.");
    end

    g = 9.81;
    f1_nominal = 1 / (2*pi*sqrt(L_m/g));
    F1_MIN = max(0.15, 0.60*f1_nominal);
    F1_MAX = min(1.40, 1.50*f1_nominal);
    F2_MIN = max(F2_MIN_BASE, 1.50*F1_MAX);
    F2_MAX_LOCAL = F2_MAX;

    Aguess = 0.5 * (max(y) - min(y));
    if ~isfinite(Aguess) || Aguess < 1e-6; Aguess = 0.1; end

    if init == 1
        f1_guess = f1_nominal;
        d1_guess = 0.005;
        phi1_guess = 0;
        A1_guess = Aguess;
    else
        f1_guess = clamp(f1_nominal * (0.80 + 0.40*rand), F1_MIN, F1_MAX);
        d1_guess = 0.001 + 0.08*rand;
        phi1_guess = -pi + 2*pi*rand;
        A1_guess = Aguess * (0.5 + 1.5*rand) * randSign();
    end

    q1_0 = [0; A1_guess; d1_guess; f1_guess; phi1_guess];
    q1_best = fminsearch(@(q) mode1Cost(q, tr, y, wFit, F1_MIN, F1_MAX, USE_QUANTIZATION_AWARE_COST, QUANTIZATION_DEADBAND_DEG), q1_0, optsFit);
    [~, ~, y_mode1_fit] = mode1Model(q1_best, tr);
    residual_after_mode1 = y - y_mode1_fit;

    [f_fft, amp_fft] = singleSidedFFT(residual_after_mode1, DT);
    band = f_fft >= F2_MIN & f_fft <= F2_MAX_LOCAL;
    if any(band)
        fBand = f_fft(band);
        aBand = amp_fft(band);
        [~, idx] = sort(aBand, "descend");
        peakFreqs = fBand(idx(1:min(8, numel(idx))));
    else
        peakFreqs = max(F2_MIN, 2.5*f1_nominal);
    end

    if init == 1
        f2_guess = peakFreqs(1);
        d2_guess = 0.8;
        phi2_guess = 0;
    else
        candidateF2 = unique([peakFreqs(:).' 2 3 4 5 6 8 10 12 15]);
        candidateF2 = candidateF2(candidateF2 >= F2_MIN & candidateF2 <= F2_MAX_LOCAL);
        if isempty(candidateF2)
            candidateF2 = clamp(2.5*f1_nominal, F2_MIN, F2_MAX_LOCAL);
        end
        f2_guess = candidateF2(randi(numel(candidateF2)));
        f2_guess = clamp(f2_guess * (0.90 + 0.20*rand), F2_MIN, F2_MAX_LOCAL);
        d2_guess = 0.05 + 3.0*rand;
        phi2_guess = -pi + 2*pi*rand;
    end

    A2_guess = 0.5 * (max(residual_after_mode1) - min(residual_after_mode1));
    if ~isfinite(A2_guess) || A2_guess < 0.01
        A2_guess = 0.05 * max(abs(Aguess), 1e-6);
    end
    if init > 1
        A2_guess = A2_guess * (0.5 + 1.8*rand) * randSign();
    end

    q2_0 = [q1_best(1); q1_best(2); max(q1_best(3), 0.001); q1_best(4); q1_best(5); A2_guess; d2_guess; f2_guess; phi2_guess];
    q2_best = fminsearch(@(q) twoModeCostConstrained(q, tr, y, wFit, F1_MIN, F1_MAX, F2_MIN, F2_MAX_LOCAL, USE_QUANTIZATION_AWARE_COST, QUANTIZATION_DEADBAND_DEG), q2_0, optsFit);

    [bias2, mode1, mode2, yFit] = twoModeModel(q2_best, tr);
    fitResidual = y - yFit;

    f1 = q2_best(4); d1 = q2_best(3);
    f2 = q2_best(8); d2 = q2_best(7);
    omega_d1 = 2*pi*f1;
    omega_n1 = sqrt(omega_d1^2 + d1^2);
    zeta1 = d1 / omega_n1;
    omega_d2 = 2*pi*f2;
    omega_n2 = sqrt(omega_d2^2 + d2^2);
    zeta2 = d2 / omega_n2;

    mode1Energy = trapz(tr, mode1.^2);
    mode2Energy = trapz(tr, mode2.^2);
    fitTotalEnergy = trapz(tr, (mode1 + mode2).^2);
    measuredTotalEnergy = trapz(tr, y.^2);
    crossEnergy = fitTotalEnergy - mode1Energy - mode2Energy;

    g = 9.80665;
    [mode1EintJkg, mode1EavgJkg] = massNormalizedEnergy(mode1, tr, L_m, g);
    [mode2EintJkg, mode2EavgJkg] = massNormalizedEnergy(mode2, tr, L_m, g);
    [fitTotalEintJkg, fitTotalEavgJkg] = massNormalizedEnergy(mode1 + mode2, tr, L_m, g);
    [measuredEintJkg, measuredEavgJkg] = massNormalizedEnergy(y, tr, L_m, g);

    mode1OnlyRmse = rms_omitnan(y - y_mode1_fit);
    twoModeRmse = rms_omitnan(fitResidual);
    weightedRawRmse = weighted_rms_omitnan(fitResidual, wFit);
    fitObjectiveCost = weighted_quantized_rms(fitResidual, wFit, USE_QUANTIZATION_AWARE_COST, QUANTIZATION_DEADBAND_DEG);

    idxEarly05 = tr <= 0.50;
    idxEarly2 = tr <= 2.00;

    m.tf_s = tf;
    m.residual_start_abs_s = tStart;
    m.residual_end_abs_s = tEnd;
    m.residual_duration_actual_s = tEnd - tStart;
    m.n_samples = numel(y);

    m.F1_MIN = F1_MIN; m.F1_MAX = F1_MAX; m.F2_MIN = F2_MIN; m.F2_MAX = F2_MAX_LOCAL;
    m.bias_deg = bias2;
    m.A1_deg = q2_best(2); m.A1_abs_deg = abs(q2_best(2));
    m.A2_deg = q2_best(6); m.A2_abs_deg = abs(q2_best(6));
    m.f1_damped_Hz = f1; m.f2_damped_Hz = f2;
    m.T1_damped_s = 1/f1; m.T2_damped_s = 1/f2;
    m.d1_1_s = d1; m.d2_1_s = d2;
    m.omega_d1_rad_s = omega_d1; m.omega_d2_rad_s = omega_d2;
    m.omega_n1_rad_s = omega_n1; m.omega_n2_rad_s = omega_n2;
    m.zeta1 = zeta1; m.zeta2 = zeta2;
    m.phi1_rad = q2_best(5); m.phi2_rad = q2_best(9);

    m.mode1_energy = mode1Energy;
    m.mode2_energy = mode2Energy;
    m.fit_total_energy = fitTotalEnergy;
    m.measured_total_energy = measuredTotalEnergy;
    m.cross_energy = crossEnergy;

    m.mode1_Eint_Js_per_kg = mode1EintJkg;
    m.mode2_Eint_Js_per_kg = mode2EintJkg;
    m.fit_total_Eint_Js_per_kg = fitTotalEintJkg;
    m.measured_Eint_Js_per_kg = measuredEintJkg;
    m.mode1_Eavg_J_per_kg = mode1EavgJkg;
    m.mode2_Eavg_J_per_kg = mode2EavgJkg;
    m.fit_total_Eavg_J_per_kg = fitTotalEavgJkg;
    m.measured_Eavg_J_per_kg = measuredEavgJkg;

    m.mode1_energy_pct_of_measured = 100 * mode1Energy / max(measuredTotalEnergy, eps);
    m.mode2_energy_pct_of_measured = 100 * mode2Energy / max(measuredTotalEnergy, eps);
    m.fit_total_energy_pct_of_measured = 100 * fitTotalEnergy / max(measuredTotalEnergy, eps);

    m.mode1_rms_deg = rms_omitnan(mode1);
    m.mode2_rms_deg = rms_omitnan(mode2);
    m.measured_rms_deg = rms_omitnan(y);
    m.mode1_p2p_deg = max(mode1) - min(mode1);
    m.mode2_p2p_deg = max(mode2) - min(mode2);
    m.measured_p2p_deg = max(y) - min(y);

    m.mode1_only_rmse_deg = mode1OnlyRmse;
    m.two_mode_rmse_deg = twoModeRmse;
    m.weighted_raw_rmse_deg = weightedRawRmse;
    m.fit_rmse_deg = twoModeRmse;
    m.fit_objective_cost = fitObjectiveCost;
    m.rmse_improvement_pct = 100 * (mode1OnlyRmse - twoModeRmse) / max(mode1OnlyRmse, eps);
    m.two_mode_rmse_first_0p5s_deg = rms_omitnan(fitResidual(idxEarly05));
    m.two_mode_rmse_first_2s_deg = rms_omitnan(fitResidual(idxEarly2));
    m.two_mode_rmse_full_10s_deg = twoModeRmse;

    m.weighted_fit_used = USE_EARLY_WEIGHTED_FIT;
    m.very_early_window_s = VERY_EARLY_WINDOW_S;
    m.early_window_s = EARLY_WINDOW_S;
    m.very_early_weight = VERY_EARLY_WEIGHT;
    m.early_weight = EARLY_WEIGHT;
    m.late_weight = LATE_WEIGHT;
    m.quantization_aware_cost_used = USE_QUANTIZATION_AWARE_COST;
    m.quantization_deg = QUANTIZATION_DEG;
    m.quantization_deadband_deg = QUANTIZATION_DEADBAND_DEG;
    m.residual_rms_over_quantization = twoModeRmse / max(QUANTIZATION_DEG, eps);

    m = applyQuantizationReportingFilter(m, ...
        MODE2_QUANT_MAX_AMP_DEG, MODE2_QUANT_MAX_P2P_DEG, MODE2_QUANT_MAX_RMS_DEG, ...
        WHOLE_RUN_QUANT_MAX_P2P_DEG, WHOLE_RUN_QUANT_MAX_RMS_DEG);

    m.f2_fft_guess_Hz = f2_guess;
    m.f2_near_lower_bound = abs(f2 - F2_MIN) < 0.03;
    m.f2_near_upper_bound = abs(f2 - F2_MAX_LOCAL) < 0.10;
    m.mode2_large_relative_to_mode1 = abs(m.A2_deg) > 1.2 * max(abs(m.A1_deg), eps);

    flag = "ok";
    if m.f2_near_lower_bound; flag = flag + "_f2_lower_bound"; end
    if m.f2_near_upper_bound; flag = flag + "_f2_upper_bound"; end
    if m.mode2_large_relative_to_mode1; flag = flag + "_mode2_large"; end
    if m.rmse_improvement_pct < 5; flag = flag + "_low_2mode_improvement"; end
    if m.residual_rms_over_quantization < 1.0; flag = flag + "_within_one_quant_tick"; end
    m.fit_quality_flag = flag;

    m.t_after_tf_s = tqAbs - tf;
    m.t_fit_local_s = tr;
    m.y_raw_deg = yRaw;
    m.y_measured_demeaned_deg = y;
    m.mode1_fit_deg = mode1;
    m.mode2_fit_deg = mode2;
    m.two_mode_fit_deg = yFit;
    m.fit_residual_deg = fitResidual;
    m.mode1_only_fit_deg = y_mode1_fit;
    m.residual_after_mode1_deg = residual_after_mode1;
    m.f_fft_Hz = f_fft;
    m.amp_fft_deg = amp_fft;
    m.init_number = init;
    m.accepted_init = init;
    m.selection_score = fitObjectiveCost;
end

function m = applyQuantizationReportingFilter(m, mode2MaxAmpDeg, mode2MaxP2PDeg, mode2MaxRmsDeg, wholeMaxP2PDeg, wholeMaxRmsDeg)
    wholeRunQuantized = m.measured_p2p_deg <= wholeMaxP2PDeg && m.measured_rms_deg <= wholeMaxRmsDeg;
    mode2TickScale = m.A2_abs_deg <= mode2MaxAmpDeg || ...
                     m.mode2_p2p_deg <= mode2MaxP2PDeg || ...
                     m.mode2_rms_deg <= mode2MaxRmsDeg;

    m.quantization_only_run = wholeRunQuantized;
    m.mode2_quant_artifact = wholeRunQuantized || mode2TickScale;

    if m.quantization_only_run
        m.mode1_Eavg_reported_J_per_kg = 0;
        m.mode2_Eavg_reported_J_per_kg = 0;
        m.fit_total_Eavg_reported_J_per_kg = 0;
        m.mode1_Eint_reported_Js_per_kg = 0;
        m.mode2_Eint_reported_Js_per_kg = 0;
        m.fit_total_Eint_reported_Js_per_kg = 0;
        m.quantization_reporting_note = sprintf( ...
            "whole residual near encoder resolution: p2p=%.4g deg rms=%.4g deg", ...
            m.measured_p2p_deg, m.measured_rms_deg);
    elseif m.mode2_quant_artifact
        m.mode1_Eavg_reported_J_per_kg = m.mode1_Eavg_J_per_kg;
        m.mode2_Eavg_reported_J_per_kg = 0;
        m.fit_total_Eavg_reported_J_per_kg = m.mode1_Eavg_J_per_kg;
        m.mode1_Eint_reported_Js_per_kg = m.mode1_Eint_Js_per_kg;
        m.mode2_Eint_reported_Js_per_kg = 0;
        m.fit_total_Eint_reported_Js_per_kg = m.mode1_Eint_Js_per_kg;
        m.quantization_reporting_note = sprintf( ...
            "mode2 treated as quantization artifact: A2=%.4g deg p2p2=%.4g deg rms2=%.4g deg", ...
            m.A2_abs_deg, m.mode2_p2p_deg, m.mode2_rms_deg);
    else
        m.mode1_Eavg_reported_J_per_kg = m.mode1_Eavg_J_per_kg;
        m.mode2_Eavg_reported_J_per_kg = m.mode2_Eavg_J_per_kg;
        m.fit_total_Eavg_reported_J_per_kg = m.fit_total_Eavg_J_per_kg;
        m.mode1_Eint_reported_Js_per_kg = m.mode1_Eint_Js_per_kg;
        m.mode2_Eint_reported_Js_per_kg = m.mode2_Eint_Js_per_kg;
        m.fit_total_Eint_reported_Js_per_kg = m.fit_total_Eint_Js_per_kg;
        m.quantization_reporting_note = "mode2 retained";
    end
end

function [Eint_Js_per_kg, Eavg_J_per_kg] = massNormalizedEnergy(thetaDeg, t, L_m, g)
    theta = deg2rad(thetaDeg(:));
    t = t(:);
    thetaDot = gradient(theta, median(diff(t)));
    E = 0.5 * L_m^2 .* thetaDot.^2 + g * L_m .* (1 - cos(theta));
    Eint_Js_per_kg = trapz(t, E);
    Eavg_J_per_kg = Eint_Js_per_kg / max(t(end) - t(1), eps);
end

function row = modalMetricsToRow(m, rawFile, info, gateS)
    row = table();
    row.raw_file = string(rawFile);
    row.run_name = string(info.run_name);
    row.length_m = info.L_m;
    row.length_label = "L" + string(round(100*info.L_m));
    row.method = string(info.method);
    row.rep = info.rep;
    row.gate_after_stop_s = gateS;
    row.accepted_init = m.accepted_init;
    row.selection_score = m.selection_score;
    row.fit_objective_cost = m.fit_objective_cost;
    row.tf_s = m.tf_s;
    row.residual_start_abs_s = m.residual_start_abs_s;
    row.residual_end_abs_s = m.residual_end_abs_s;
    row.residual_duration_actual_s = m.residual_duration_actual_s;
    row.n_samples = m.n_samples;
    row.bias_deg = m.bias_deg;
    row.A1_deg = m.A1_deg; row.A1_abs_deg = m.A1_abs_deg;
    row.A2_deg = m.A2_deg; row.A2_abs_deg = m.A2_abs_deg;
    row.f1_damped_Hz = m.f1_damped_Hz; row.f2_damped_Hz = m.f2_damped_Hz;
    row.T1_damped_s = m.T1_damped_s; row.T2_damped_s = m.T2_damped_s;
    row.d1_1_s = m.d1_1_s; row.d2_1_s = m.d2_1_s;
    row.omega_d1_rad_s = m.omega_d1_rad_s; row.omega_d2_rad_s = m.omega_d2_rad_s;
    row.omega_n1_rad_s = m.omega_n1_rad_s; row.omega_n2_rad_s = m.omega_n2_rad_s;
    row.zeta1 = m.zeta1; row.zeta2 = m.zeta2;
    row.phi1_rad = m.phi1_rad; row.phi2_rad = m.phi2_rad;
    row.mode1_energy = m.mode1_energy; row.mode2_energy = m.mode2_energy;
    row.mode1_Eavg_J_per_kg = m.mode1_Eavg_J_per_kg;
    row.mode2_Eavg_J_per_kg = m.mode2_Eavg_J_per_kg;
    row.fit_total_Eavg_J_per_kg = m.fit_total_Eavg_J_per_kg;
    row.measured_Eavg_J_per_kg = m.measured_Eavg_J_per_kg;
    row.mode1_Eint_Js_per_kg = m.mode1_Eint_Js_per_kg;
    row.mode2_Eint_Js_per_kg = m.mode2_Eint_Js_per_kg;
    row.fit_total_Eint_Js_per_kg = m.fit_total_Eint_Js_per_kg;
    row.measured_Eint_Js_per_kg = m.measured_Eint_Js_per_kg;
    row.mode1_Eavg_reported_J_per_kg = m.mode1_Eavg_reported_J_per_kg;
    row.mode2_Eavg_reported_J_per_kg = m.mode2_Eavg_reported_J_per_kg;
    row.fit_total_Eavg_reported_J_per_kg = m.fit_total_Eavg_reported_J_per_kg;
    row.mode1_Eint_reported_Js_per_kg = m.mode1_Eint_reported_Js_per_kg;
    row.mode2_Eint_reported_Js_per_kg = m.mode2_Eint_reported_Js_per_kg;
    row.fit_total_Eint_reported_Js_per_kg = m.fit_total_Eint_reported_Js_per_kg;
    row.quantization_only_run = m.quantization_only_run;
    row.mode2_quant_artifact = m.mode2_quant_artifact;
    row.quantization_reporting_note = string(m.quantization_reporting_note);
    row.fit_total_energy = m.fit_total_energy; row.measured_total_energy = m.measured_total_energy;
    row.cross_energy = m.cross_energy;
    row.mode1_energy_pct_of_measured = m.mode1_energy_pct_of_measured;
    row.mode2_energy_pct_of_measured = m.mode2_energy_pct_of_measured;
    row.fit_total_energy_pct_of_measured = m.fit_total_energy_pct_of_measured;
    row.mode1_rms_deg = m.mode1_rms_deg; row.mode2_rms_deg = m.mode2_rms_deg; row.measured_rms_deg = m.measured_rms_deg;
    row.mode1_p2p_deg = m.mode1_p2p_deg; row.mode2_p2p_deg = m.mode2_p2p_deg; row.measured_p2p_deg = m.measured_p2p_deg;
    row.mode1_only_rmse_deg = m.mode1_only_rmse_deg;
    row.two_mode_rmse_deg = m.two_mode_rmse_deg;
    row.weighted_raw_rmse_deg = m.weighted_raw_rmse_deg;
    row.fit_rmse_deg = m.fit_rmse_deg;
    row.rmse_improvement_pct = m.rmse_improvement_pct;
    row.two_mode_rmse_first_0p5s_deg = m.two_mode_rmse_first_0p5s_deg;
    row.two_mode_rmse_first_2s_deg = m.two_mode_rmse_first_2s_deg;
    row.two_mode_rmse_full_10s_deg = m.two_mode_rmse_full_10s_deg;
    row.weighted_fit_used = m.weighted_fit_used;
    row.very_early_window_s = m.very_early_window_s;
    row.early_window_s = m.early_window_s;
    row.very_early_weight = m.very_early_weight;
    row.early_weight = m.early_weight;
    row.late_weight = m.late_weight;
    row.quantization_aware_cost_used = m.quantization_aware_cost_used;
    row.quantization_deg = m.quantization_deg;
    row.quantization_deadband_deg = m.quantization_deadband_deg;
    row.residual_rms_over_quantization = m.residual_rms_over_quantization;
    row.f2_fft_guess_Hz = m.f2_fft_guess_Hz;
    row.f2_near_lower_bound = m.f2_near_lower_bound;
    row.f2_near_upper_bound = m.f2_near_upper_bound;
    row.mode2_large_relative_to_mode1 = m.mode2_large_relative_to_mode1;
    row.fit_quality_flag = string(m.fit_quality_flag);
end

function row = initializationLogRow(m, rawFile, info, gateS, init, success, reason)
    row = table();
    row.raw_file = string(rawFile); row.run_name = string(info.run_name); row.length_m = info.L_m;
    row.method = string(info.method); row.rep = info.rep; row.gate_after_stop_s = gateS;
    row.init_number = init; row.success = success; row.reason = string(reason);
    row.selection_score = m.selection_score; row.fit_objective_cost = m.fit_objective_cost; row.weighted_raw_rmse_deg = m.weighted_raw_rmse_deg;
    row.two_mode_rmse_deg = m.two_mode_rmse_deg; row.two_mode_rmse_first_0p5s_deg = m.two_mode_rmse_first_0p5s_deg;
    row.two_mode_rmse_first_2s_deg = m.two_mode_rmse_first_2s_deg; row.mode1_only_rmse_deg = m.mode1_only_rmse_deg;
    row.rmse_improvement_pct = m.rmse_improvement_pct;
    row.A1_abs_deg = m.A1_abs_deg; row.f1_damped_Hz = m.f1_damped_Hz; row.zeta1 = m.zeta1; row.mode1_energy = m.mode1_energy;
    row.A2_abs_deg = m.A2_abs_deg; row.f2_damped_Hz = m.f2_damped_Hz; row.zeta2 = m.zeta2; row.mode2_energy = m.mode2_energy;
    row.measured_total_energy = m.measured_total_energy; row.residual_rms_over_quantization = m.residual_rms_over_quantization;
    row.fit_quality_flag = string(m.fit_quality_flag);
end

function row = failedInitializationLogRow(rawFile, info, gateS, init, reason)
    row = table();
    row.raw_file = string(rawFile); row.run_name = string(info.run_name); row.length_m = info.L_m;
    row.method = string(info.method); row.rep = info.rep; row.gate_after_stop_s = gateS;
    row.init_number = init; row.success = false; row.reason = string(reason);
    row.selection_score = inf; row.fit_objective_cost = inf; row.weighted_raw_rmse_deg = NaN;
    row.two_mode_rmse_deg = NaN; row.two_mode_rmse_first_0p5s_deg = NaN; row.two_mode_rmse_first_2s_deg = NaN;
    row.mode1_only_rmse_deg = NaN; row.rmse_improvement_pct = NaN;
    row.A1_abs_deg = NaN; row.f1_damped_Hz = NaN; row.zeta1 = NaN; row.mode1_energy = NaN;
    row.A2_abs_deg = NaN; row.f2_damped_Hz = NaN; row.zeta2 = NaN; row.mode2_energy = NaN;
    row.measured_total_energy = NaN; row.residual_rms_over_quantization = NaN; row.fit_quality_flag = "failed";
end

function saveBestFitDecomposition(m, rawFile, info, gateS, FIT_DIR)
    n = numel(m.t_after_tf_s);
    out = table();
    out.raw_file = repmat(string(rawFile), n, 1);
    out.run_name = repmat(string(info.run_name), n, 1);
    out.length_m = repmat(info.L_m, n, 1);
    out.method = repmat(string(info.method), n, 1);
    out.rep = repmat(info.rep, n, 1);
    out.gate_after_stop_s = repmat(gateS, n, 1);
    out.accepted_init = repmat(m.accepted_init, n, 1);
    out.selection_score = repmat(m.selection_score, n, 1);
    out.fit_objective_cost = repmat(m.fit_objective_cost, n, 1);
    out.weighted_raw_rmse_deg = repmat(m.weighted_raw_rmse_deg, n, 1);
    out.two_mode_rmse_deg = repmat(m.two_mode_rmse_deg, n, 1);
    out.quantization_deg = repmat(m.quantization_deg, n, 1);
    out.quantization_deadband_deg = repmat(m.quantization_deadband_deg, n, 1);
    out.t_after_tf_s = m.t_after_tf_s;
    out.t_fit_local_s = m.t_fit_local_s;
    out.y_raw_deg = m.y_raw_deg;
    out.y_measured_demeaned_deg = m.y_measured_demeaned_deg;
    out.mode1_fit_deg = m.mode1_fit_deg;
    out.mode2_fit_deg = m.mode2_fit_deg;
    out.two_mode_fit_deg = m.two_mode_fit_deg;
    out.fit_residual_deg = m.fit_residual_deg;
    out.mode1_only_fit_deg = m.mode1_only_fit_deg;
    out.residual_after_mode1_deg = m.residual_after_mode1_deg;
    gateTag = sprintf("gate_%04dms", round(1000*gateS));
    outName = sprintf("bestfit_%s_L%03d_%s_rep%02d_init%02d.csv", gateTag, round(100*info.L_m), sanitizeFileName(info.method), info.rep, m.accepted_init);
    writetable(out, fullfile(FIT_DIR, outName));
end

function cost = mode1Cost(q, t, y, wFit, f1min, f1max, USE_QUANTIZATION_AWARE_COST, QUANTIZATION_DEADBAND_DEG)
    bias = q(1); A1 = q(2); d1 = q(3); f1 = q(4);
    if abs(bias) > 10 || abs(A1) > 80 || d1 < 0 || d1 > 2 || f1 < f1min || f1 > f1max
        cost = 1e6; return;
    end
    [~, ~, yFit] = mode1Model(q, t);
    err = y - yFit;
    cost = weighted_quantized_rms(err, wFit, USE_QUANTIZATION_AWARE_COST, QUANTIZATION_DEADBAND_DEG);
end

function [bias, mode1, yFit] = mode1Model(q, t)
    bias = q(1); A1 = q(2); d1 = q(3); f1 = q(4); phi1 = q(5);
    mode1 = A1 .* exp(-d1 .* t) .* sin(2*pi*f1.*t + phi1);
    yFit = bias + mode1;
end

function cost = twoModeCostConstrained(q, t, y, wFit, f1min, f1max, f2min, f2max, USE_QUANTIZATION_AWARE_COST, QUANTIZATION_DEADBAND_DEG)
    bias = q(1); A1 = q(2); d1 = q(3); f1 = q(4); A2 = q(6); d2 = q(7); f2 = q(8);
    if abs(bias) > 10 || abs(A1) > 80 || abs(A2) > 80 || d1 < 0 || d1 > 2 || d2 < 0 || d2 > 40 || f1 < f1min || f1 > f1max || f2 < f2min || f2 > f2max
        cost = 1e6; return;
    end
    [~, ~, ~, yFit] = twoModeModel(q, t);
    err = y - yFit;
    cost = weighted_quantized_rms(err, wFit, USE_QUANTIZATION_AWARE_COST, QUANTIZATION_DEADBAND_DEG);
    if abs(A2) > 1.8 * max(abs(A1), eps)
        cost = cost + 0.03 * (abs(A2) - 1.8*abs(A1));
    end
end

function [bias, mode1, mode2, yFit] = twoModeModel(q, t)
    bias = q(1);
    A1 = q(2); d1 = q(3); f1 = q(4); phi1 = q(5);
    A2 = q(6); d2 = q(7); f2 = q(8); phi2 = q(9);
    mode1 = A1 .* exp(-d1 .* t) .* sin(2*pi*f1.*t + phi1);
    mode2 = A2 .* exp(-d2 .* t) .* sin(2*pi*f2.*t + phi2);
    yFit = bias + mode1 + mode2;
end

function [f, amp] = singleSidedFFT(y, dt)
    y = y(:); y = y - mean_omitnan(y);
    N = numel(y);
    n = (0:N-1).';
    w = 0.5 * (1 - cos(2*pi*n/(N-1)));
    yw = y .* w;
    Y = fft(yw);
    P2 = abs(Y/N);
    P1 = P2(1:floor(N/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    Fs = 1/dt;
    f = Fs*(0:floor(N/2)).'/N;
    amp = P1;
end

function r = weighted_quantized_rms(err, w, useQuantAware, qDeadband)
    err = err(:); w = w(:);
    valid = isfinite(err) & isfinite(w) & w > 0;
    err = err(valid); w = w(valid);
    if isempty(err); r = NaN; return; end
    if useQuantAware
        eEff = max(abs(err) - qDeadband, 0);
    else
        eEff = abs(err);
    end
    r = sqrt(sum(w .* eEff.^2) / sum(w));
end

function r = weighted_rms_omitnan(x, w)
    x = x(:); w = w(:);
    valid = isfinite(x) & isfinite(w) & w > 0;
    x = x(valid); w = w(valid);
    if isempty(x)
        r = NaN;
    else
        r = sqrt(sum(w .* x.^2) / sum(w));
    end
end

function failed = appendFail(failed, rawFile, reason)
    row = table(); row.raw_file = string(rawFile); row.reason = string(reason);
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
        if cmdMag(i) > thresh; nonzeroSeen = true; end
        if nonzeroSeen && cmdMag(i) <= thresh
            tf = t(i); return;
        end
    end
    error("Could not detect stop time.");
end

function [angle_deg, source] = getAngleSignalDeg(T, names, L_m, swing_mm)
    if any(names == "swing_pitch_deg")
        angle_deg = forceNumeric(T.("swing_pitch_deg")); source = "swing_pitch_deg"; return;
    end
    if any(names == "swing_axis_angle_deg")
        angle_deg = forceNumeric(T.("swing_axis_angle_deg")); source = "swing_axis_angle_deg"; return;
    end
    if any(names == "enc_pitch_deg")
        angle_deg = forceNumeric(T.("enc_pitch_deg")); source = "enc_pitch_deg"; return;
    end
    if any(names == "payload_pitch_deg")
        angle_deg = forceNumeric(T.("payload_pitch_deg")); source = "payload_pitch_deg"; return;
    end
    if any(names == "enc_roll_deg")
        angle_deg = forceNumeric(T.("enc_roll_deg")); source = "enc_roll_deg"; return;
    end
    if any(isfinite(swing_mm)) && isfinite(L_m)
        angle_rad = (swing_mm ./ 1000) ./ L_m;
        angle_deg = rad2deg(angle_rad);
        source = "swing_mm_converted_to_deg"; return;
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
            x = forceNumeric(T.(c)); return;
        end
    end
end

function x = forceNumeric(x)
    if isnumeric(x) || islogical(x)
        x = double(x); x = x(:); return;
    end
    if iscell(x); x = string(x); end
    if isstring(x) || ischar(x) || iscategorical(x)
        x = str2double(string(x)); x = x(:); return;
    end
    error("Invalid data type.");
end

function x = resizeToN(x, n)
    x = x(:);
    if numel(x) == n; return; end
    if isempty(x)
        x = nan(n,1); return;
    end
    if numel(x) > n
        x = x(1:n);
    else
        x(end+1:n,1) = NaN;
    end
end

function T = safeAppendTable(T, row)
    if isempty(T)
        T = row;
    else
        T = [T; row];
    end
end

function val = clamp(val, lo, hi)
    val = max(lo, min(hi, val));
end

function s = randSign()
    if rand < 0.5; s = -1; else; s = 1; end
end

function s = sanitizeFileName(s)
    s = string(s);
    s = regexprep(s, "[^\w\-\.]", "_");
end

function r = rms_omitnan(x)
    x = x(:); x = x(isfinite(x));
    if isempty(x)
        r = NaN;
    else
        r = sqrt(mean(x.^2));
    end
end

function m = mean_omitnan(x)
    x = x(:); x = x(isfinite(x));
    if isempty(x)
        m = NaN;
    else
        m = mean(x);
    end
end
