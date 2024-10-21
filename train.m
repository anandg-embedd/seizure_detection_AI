% Load the dataset
file_path = 'dataset/eeg-seizure_train.npz';
data = load(file_path);

train_signals = data.train_signals;
train_labels = data.train_labels;

% Log the shape of the loaded data
disp(['Shape of train_signals: ', num2str(size(train_signals))]);
disp(['Shape of train_labels: ', num2str(size(train_labels))]);

% Reduce the dataset size by taking a subset (e.g., 10%)
subset_size = floor(0.1 * size(train_signals, 1));
train_signals = train_signals(1:subset_size, :, :);
train_labels = train_labels(1:subset_size);

% Log the shape after taking a subset
disp(['Shape of train_signals after subset: ', num2str(size(train_signals))]);
disp(['Shape of train_labels after subset: ', num2str(size(train_labels))]);

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

% Extract DWT features from the signals
train_features = extract_dwt_features(train_signals);

% Log the shape of the extracted features
disp(['Shape of extracted features: ', num2str(size(train_features))]);

% Split the data into training, validation, and test sets
cv = cvpartition(size(train_features, 1), 'HoldOut', 0.3);
X_train = train_features(training(cv), :);
y_train = train_labels(training(cv));
X_temp = train_features(test(cv), :);
y_temp = train_labels(test(cv));

cv_temp = cvpartition(size(X_temp, 1), 'HoldOut', 0.5);
X_val = X_temp(training(cv_temp), :);
y_val = y_temp(training(cv_temp));
X_test = X_temp(test(cv_temp), :);
y_test = y_temp(test(cv_temp));

% Log the shapes after splitting the data
disp(['Shape of X_train: ', num2str(size(X_train))]);
disp(['Shape of X_val: ', num2str(size(X_val))]);
disp(['Shape of X_test: ', num2str(size(X_test))]);
disp(['Shape of y_train: ', num2str(size(y_train))]);
disp(['Shape of y_val: ', num2str(size(y_val))]);
disp(['Shape of y_test: ', num2str(size(y_test))]);

% Reshape the data for the RNN
X_train = reshape(X_train, [size(X_train, 1), size(X_train, 2), 1]);
X_val = reshape(X_val, [size(X_val, 1), size(X_val, 2), 1]);
X_test = reshape(X_test, [size(X_test, 1), size(X_test, 2), 1]);

% Log the shapes after reshaping
disp(['Shape of X_train after reshaping: ', num2str(size(X_train))]);
disp(['Shape of X_val after reshaping: ', num2str(size(X_val))]);
disp(['Shape of X_test after reshaping: ', num2str(size(X_test))]);

% Build the improved RNN model with Conv1D layers and additional LSTM layers
layers = [
    sequenceInputLayer([size(X_train, 2) 1])
    convolution1dLayer(3, 64, 'Padding', 'same')
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2)
    batchNormalizationLayer
    convolution1dLayer(3, 128, 'Padding', 'same')
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2)
    batchNormalizationLayer
    lstmLayer(100, 'OutputMode', 'last')
    fullyConnectedLayer(1)
    sigmoidLayer
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {X_val, y_val}, ...
    'Plots', 'training-progress');

% Train the model
net = trainNetwork(X_train, y_train, layers, options);

% Evaluate the model on the test set
YPred = predict(net, X_test);
test_accuracy = sum(round(YPred) == y_test) / numel(y_test);
disp(['Test Accuracy: ', num2str(test_accuracy)]);

% Save the model
save('seizure_detection_model.mat', 'net');
disp('Model training complete and saved as seizure_detection_model.mat');
