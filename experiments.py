from mnist_classifier import MnistClassifier

class Experiments:

    def get_models(self):
        mc = MnistClassifier()

        model_name = ['hlu-128_hla-relu_co-adam_cl-categorical_crossentropy_fe-10_fbs-128', 'hlu-128_hla-relu_co-adam_cl-categorical_crossentropy_fe-20_fbs-128', 'hlu-128_hla-relu_co-adam_cl-categorical_crossentropy_fe-30_fbs-128']
        hidden_layer_units = [128, 128, 128]
        hidden_layer_activation = ['relu', 'relu', 'relu']
        compile_optimizer = ['adam', 'adam', 'adam']
        compile_loss = ['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy']
        fit_epochs = [10, 20, 30]
        fit_batch_size = [128, 128, 128]

        for i in range(len(model_name)):
            mc.train_model(model_name[i], hidden_layer_units[i], hidden_layer_activation[i], compile_optimizer[i], compile_loss[i], fit_epochs[i], fit_batch_size[i])

if __name__ == "__main__":
    experiments = Experiments()
    experiments.get_models()