from lightgbm import LGBMClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def add_noise(x, noise_std=0.03):
    return x + np.random.normal(0, noise_std, size=x.shape)

def sliding_window_augment(data, labels, shifts=None, noise_prob=0.35, noise_std=0.05):
    if shifts is None:
        shifts = [0]
    N = data.shape[0]
    aug_data = [data]
    aug_labels = [labels]
    
    for shift in shifts:
        new_samples = []
        new_labels = []
        for i in range(N - 1):
            if labels[i] == labels[i + 1]:
                new_window = np.concatenate([
                    data[i, shift:, :],
                    data[i + 1, :shift, :]
                ], axis=0)

                if np.random.rand() < noise_prob:
                    new_window = add_noise(new_window, noise_std)
                new_samples.append(new_window)
                new_labels.append(labels[i])
        
        if new_samples:
            aug_data.append(np.stack(new_samples, axis=0))
            aug_labels.append(np.array(new_labels))
    
    return np.concatenate(aug_data, axis=0), np.concatenate(aug_labels)

def extract_features_single(signal_60x6):
    feats = []
    for ch in range(6):
        x = signal_60x6[:, ch]

        ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        feats.append(0.0 if np.isnan(ac1) else ac1)
        feats.append(np.polyfit(np.arange(len(x)), x, 1)[0])

        x = (x - np.mean(x)) / (np.std(x) + 1e-6)

        feats += [
            np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x),
            np.percentile(x, 75) - np.percentile(x, 25),
            np.sum(x**2),
            ((x[:-1] * x[1:]) < 0).sum(),
            np.sqrt(np.mean(x**2)),
            np.mean(np.abs(np.diff(x))),
            np.max(x) - np.min(x),
            np.mean(np.abs(x)),
            np.var(x),
            np.percentile(x, 10),
            np.percentile(x, 90),
        ]

        Xf = np.abs(np.fft.rfft(x))
        Xf_total = np.sum(Xf) + 1e-12
        Xf_norm = Xf / Xf_total
        freqs = np.fft.rfftfreq(len(x))

        feats.append(np.sum(Xf_norm ** 2))  # Energy concentration
        feats.append(np.sum(freqs * Xf) / Xf_total)  # Spectral centroid
        feats.append(-np.sum(Xf_norm * np.log(Xf_norm + 1e-12)))  # Entropy

        geom_mean = np.exp(np.mean(np.log(Xf + 1e-12)))
        feats.append(geom_mean / (np.mean(Xf) + 1e-12))  # Flatness

        centroid = np.sum(freqs * Xf) / Xf_total
        feats.append(np.sqrt(np.sum(((freqs - centroid) ** 2) * Xf) / Xf_total))  # Bandwidth

    return np.array(feats, dtype=np.float32)

def extract_features_batch(data):
    return np.vstack([extract_features_single(x) for x in data])

def temporal_smoothing(predictions, window_size=5):
    smoothed = median_filter(predictions, size=window_size)
    return smoothed.astype(predictions.dtype)

def predict_test(train_data, train_labels, test_data):
    np.random.seed(42)

    train_data_aug, train_labels_aug = sliding_window_augment(
        train_data, train_labels, shifts=list(range(0, 30, 3)), noise_prob=0.5, noise_std=0.03
    )
    y_train = train_labels_aug - 1
    X_train = extract_features_batch(train_data_aug)
    X_test = extract_features_batch(test_data)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    clf = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        boosting_type="gbdt",
        learning_rate=0.07760771108377519,
        n_estimators=483,
        max_depth=9,
        num_leaves=111,
        min_child_samples=186,
        min_child_weight=0.004813353392916629,
        reg_alpha=0.0001110597623438864,
        reg_lambda=0.0032728933713887624,
        verbosity=-1,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    
    pred = clf.predict(X_test)
    pred_smoothed = temporal_smoothing(pred, window_size=3)
    pred_final = pred_smoothed + 1
    
    return pred_final
