from tensorflow.keras.datasets import mnist
import numpy as np

class MNISTDataProcessor:
    def __init__(self):
        (self.x_train_raw, self.y_train), (self.x_test_raw, self.y_test) = mnist.load_data()
        self.num_train = self.x_train_raw.shape[0]
        self.num_test = self.x_test_raw.shape[0]
        self.x_train_ft = np.fft.fftshift(np.fft.fft2(self.x_train_raw), axes=(1, 2))
        self.x_test_ft = np.fft.fftshift(np.fft.fft2(self.x_test_raw), axes=(1, 2))
    
    def norm_inputs(self, inputs, feature_axis=1):
        # if feature_axis == 1:
        #     n_features, n_examples = inputs.shape
        # elif feature_axis == 0:
        #     n_examples, n_features = inputs.shape
        # for i in range(n_features):
        #     # this is normalization along the features?
        #     l1_norm = np.mean(np.abs(inputs[i, :]))
        #     inputs[i, :] /= l1_norm

        # return inputs

        if feature_axis == 1:
            n_examples, n_features = inputs.shape
        elif feature_axis == 0:
            n_features, n_examples = inputs.shape
        # return inputs / np.linalg.norm(inputs,
        #                                axis=feature_axis).reshape(-1,1) * 16
        return inputs / np.linalg.norm(inputs,
                                       ord=1,
                                       axis=feature_axis).reshape(-1,1)
    # we use fourier orders for data processing such that the required ports number is less.
    def fourier(self, freq_radius):
        min_r, max_r = 14 - freq_radius, 14 + freq_radius

        x_train_ft = self.x_train_ft[:, min_r:max_r, min_r:max_r]
        x_test_ft = self.x_test_ft[:, min_r:max_r, min_r:max_r]
        x_train=self.norm_inputs(x_train_ft.reshape((self.num_train, -1)))
        y_train=np.eye(10)[self.y_train]
        x_test=self.norm_inputs(x_test_ft.reshape((self.num_test, -1)))
        y_test=np.eye(10)[self.y_test]
        return (np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test))

