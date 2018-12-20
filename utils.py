import os
import torch


def save_model(rnn, epoch, save_dir):
    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(epoch))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    state = rnn.state_dict()
    torch.save(state, file_path)
    print('Model saved to {}'.format(file_path))


def make_log(train_loss, test_loss, epoch, log_dir):
    file_path = os.path.join(log_dir, "epoch_" + str(epoch) + ".txt")
    train_log = "Train loss at epoch " + str(epoch) + ": " + str(train_loss) + "\n"
    test_log = "Test loss at epoch " + str(epoch) + ": " + str(test_loss)
    with open(file_path, "w") as f:
        f.write(train_log)
        f.write(test_log)
    print('Log made to {}'.format(file_path))