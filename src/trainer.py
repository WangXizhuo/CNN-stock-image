from img_data.src.model import *
from torch.utils.data import DataLoader, random_split

# # recommended only parameters
#
# torch.save(model.state_dict(), "model.pth")
# model = ConvNet_20day()
# model.load_state_dict(torch.load("model.pth"))
# model =
TRAIN_YEAR_START = 1993
TRAIN_YEAR_END = 1993
TEST_YEAR_START = 1993
TEST_YEAR_END = 1993
IMAGE_DIR = './monthly_20d'
LOG_DIR = "./loss_train"

if __name__ == '__main__':
    batch_size = 128
    learning_rate = 1e-5
    epochs = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available(), flush=True)
    print(f'Using {device}!')

    model = ConvNet_20day().to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    loss_function = loss_function.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_val_set = StockImage(IMAGE_DIR, TRAIN_YEAR_START, TRAIN_YEAR_END, transforms)

    train_set_size = int(len(train_val_set) * 0.7)
    val_set_size = len(train_val_set) - train_set_size
    train_set, val_set = random_split(train_val_set, [train_set_size, val_set_size])
    test_set = StockImage(IMAGE_DIR, TEST_YEAR_START, TEST_YEAR_END, transforms)



    print("Training data length: {};".format(len(train_set)))
    print("Validation data length: {};".format(len(val_set)))
    print("Testing data length: {};".format(len(test_set)))

    training_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    testing_data = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    total_train_step = 0
    # record logs
    writer = SummaryWriter(LOG_DIR)

    for i in range(epochs):
        print(f"--------------{i+1}th epoch starts--------------------")
        for batch, (images, labels) in enumerate(training_data):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # optimize the model
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step += 1
            if total_train_step % 50 == 0:
                print(f"loss: {loss.item()}, {total_train_step*128}/{train_set_size}")
                writer.add_scalar("train_loss", loss.item(), total_train_step)


        # check loss on validation set
        total_loss = 0.0
        with torch.no_grad():
            for batch, (images, labels) in enumerate(validation_data):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                total_loss += loss

            print(f"Total loss on validation set {total_loss/int(val_set_size/128)}.")
            writer.add_scalar("validation_loss", total_loss, i)

        torch.save(model.state_dict(), f'Conv20days_epoch{i}.pth')
        print("Model saved!")
    writer.close()

    # calculate accuracy on test set
    correct = 0

    with torch.no_grad():
        for batch, (images, labels) in enumerate(testing_data):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels.argmax(1)).sum().item()


    print(f"Total accuracy on testing dataset: {correct/len(test_set)}")
    # tensorboard --logdir=loss_train







