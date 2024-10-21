% Load the trained model
load('seizure_detection_model.mat', 'net');

% Function to extract features using Discrete Wavelet Transform (DWT)
function features = extract_dwt_features(signals)
    features = [];
    for i = 1:size(signals, 1)
        signal_features = [];
        for j = 1:size(signals, 2)
            [C, L] = wavedec(signals(i, j, :), 4, 'db4');
            coeffs_flattened = reshape(C, 1, []);
            signal_features = [signal_features; coeffs_flattened];
        end
        features = [features; signal_features];
    end
end

% Function to load a signal from a CSV file and reshape it for prediction
function signal = load_signal_from_csv(file_path)
    signal_df = readtable(file_path);
    signal = table2array(signal_df);
    signal = reshape(signal, [1, size(signal, 1), size(signal, 2)]);
end

% Function to predict seizure or normal from a CSV file
function prediction = predict_seizure(file_path, net)
    signal = load_signal_from_csv(file_path);
    disp(['Shape of raw test: ', num2str(size(signal))]);
    signal_f = extract_dwt_features(signal);
    disp(['Shape of feature test: ', num2str(size(signal_f))]);
    prediction = predict(net, signal_f);
    if prediction > 0.5
        prediction = 'Seizure';
    else
        prediction = 'Normal';
    end
end

% Example usage
csv_file = 'dataset/signal_1_label_0.csv';
prediction = predict_seizure(csv_file, net);
disp(['The prediction for ', csv_file, ' is: ', prediction]);
