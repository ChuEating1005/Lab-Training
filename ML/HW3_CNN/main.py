import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from dataset import FoodDataset, train_tfm, test_tfm
from models import get_model
from cross_validation import cross_validation, ensemble_models
from tqdm.auto import tqdm

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

def train(args):
    # Initialize model based on argument
    model, _exp_name = get_model(args.model)
    model = model.to(device)

    # Use user-provided hyperparameters
    batch_size = args.batch_size
    n_epochs = args.epochs

    # Create TensorBoard writer
    train_writer = SummaryWriter(f'runs/exp3_{_exp_name}/train')
    valid_writer = SummaryWriter(f'runs/exp3_{_exp_name}/valid')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Load data
    train_set = FoodDataset("./data/train", tfm=train_tfm)
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=6, 
        pin_memory=True,
        persistent_workers=True
    )
    valid_set = FoodDataset("./data/valid", tfm=test_tfm)
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2, 
        pin_memory=True,
        persistent_workers=True
    )

    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_acc = 0

    for epoch in range(n_epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()
            #print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)


        scheduler.step()

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):

            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            #imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                valid_pred = model(imgs.to(device))
            
            tta_preds = []
            for _ in range(3):
                with torch.no_grad():
                    aug_imgs = train_tfm(imgs).to(device)
                    pred = model(aug_imgs)
                    tta_preds.append(pred)
            
            # Integrate prediction results
            avg_tta_pred = torch.mean(torch.stack(tta_preds), dim=0)
            final_pred = avg_tta_pred * 0.2 + valid_pred * 0.8  # Use the same weight

            # We can still compute the loss (but not the gradient).
            loss = criterion(final_pred, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (final_pred.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            #break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Record to TensorBoard
        train_writer.add_scalar('Loss', train_loss, epoch)
        valid_writer.add_scalar('Loss', valid_loss, epoch)
        train_writer.add_scalar('Accuracy', train_acc, epoch)
        valid_writer.add_scalar('Accuracy', valid_acc, epoch)
        
        # Can add more information
        train_writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        valid_writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # update logs
        if valid_acc > best_acc:
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        

    train_writer.close()
    valid_writer.close()

def test(args=None):
    """
    Test the model on the test dataset
    
    Args:
        args: Command line arguments
        model: Model to test (if None, load from checkpoint)
        _exp_name: Experiment name (if None, use args.model)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If args are not provided, use global variables
    if args is None:
        batch_size = globals().get('batch_size', 64)
    else:
        batch_size = args.batch_size
    
    # If model and experiment name are not provided, load them
    model, _exp_name = get_model(args.model)
    model = model.to(device)
    model.eval()
    
    # Load test data
    test_set = FoodDataset("./data/test", tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)

    # Make predictions
    prediction = []
    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            test_tfm_pred = model(data.to(device))
            tta_preds = []
            for _ in range(5):
                with torch.no_grad():
                    aug_imgs = train_tfm(data).to(device)
                    pred = model(aug_imgs)
                    tta_preds.append(pred)
            avg_tta_pred = torch.mean(torch.stack(tta_preds), dim=0)
            final_pred = avg_tta_pred * 0.2 + test_tfm_pred * 0.8
            test_label = np.argmax(final_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()

    # Create submission file
    def pad4(i):
        return "0"*(4-len(str(i)))+str(i)
    
    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(len(test_set))]
    df["Category"] = prediction
    df.to_csv("submission.csv", index=False)
    
    return prediction

# python main.py --epochs 300 --cv --threads 4 --workers 1
def parse_args():
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--model', type=str, default='default',
                      choices=['default', 'resnet', 'mobilenet', 'efficientnet'],
                      help='model architecture to use')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0003,
                      help='learning rate')
    parser.add_argument('--seed', type=int, default=6666,
                      help='random seed')
    parser.add_argument('--workers', type=int, default=6,
                      help='number of data loading workers')
    parser.add_argument('--cv', action='store_true',
                      help='use cross-validation')
    parser.add_argument('--folds', type=int, default=4,
                      help='number of folds for cross-validation')
    parser.add_argument('--patience', type=int, default=15,
                      help='number of patience for early stopping')
    parser.add_argument('--threads', type=int, default=1,
                      help='number of parallel threads for cross-validation')
    
    # Ensemble related arguments
    parser.add_argument('--ensemble', action='store_true',
                      help='use ensemble inference')
    parser.add_argument('--voting', type=str, default='soft',
                      choices=['soft', 'hard'],
                      help='voting method for ensemble (soft: average probabilities, hard: majority vote)')
    parser.add_argument('--ensemble-models', type=str, default=None,
                      help='comma-separated list of model paths to use for ensemble')
    parser.add_argument('--ensemble-name', type=str, default=None,
                      help='name for the ensemble submission file')

    # Test related arguments
    parser.add_argument('--test', action='store_true',
                      help='test the model')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Use user-provided hyperparameters
    batch_size = args.batch_size
    n_epochs = args.epochs
    myseed = args.seed
    
    # Reset random seed
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
    
    if args.ensemble:
        print(f"Using ensemble with {args.voting} voting")
        ensemble_models(args)
    else:
        print(f"Training with {args.model} model")
        
        # Based on the argument, decide whether to use cross-validation
        if args.cv:
            print(f"Using {args.folds}-fold cross-validation")
            model = cross_validation(args)
            ensemble_models(args)
        elif args.test:
            print("Testing the model")
            test(args)
        else:
            print("Using standard train/validation split")
            train(args)
            test(args)