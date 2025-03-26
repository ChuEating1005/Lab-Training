import numpy as np
import torch
import torch.nn as nn
import concurrent.futures
import os, glob
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from models import get_model
from dataset import FoodDataset, train_tfm, test_tfm
import pandas as pd

# Function to train a single fold
def train_fold(model, train_loader, valid_loader, fold_idx, args, fold_exp_name):
    # Set model name, including fold information
    fold_exp_name = f"{fold_exp_name}/fold{fold_idx}"

    # Get the device that the model is on
    model_device = next(model.parameters()).device

    # Create TensorBoard writer
    train_writer = SummaryWriter(f'runs/{fold_exp_name}/train')
    valid_writer = SummaryWriter(f'runs/{fold_exp_name}/valid')
    
    # Use user-provided hyperparameters
    n_epochs = args.epochs

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Initialize trackers
    stale = 0
    best_acc = 0
    
    print(f"=== Training Fold {fold_idx} on {model_device} ===")

    for epoch in range(n_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            imgs, labels = batch
            
            # Forward pass - send to model's device instead of global device
            logits = model(imgs.to(model_device))
            loss = criterion(logits, labels.to(model_device))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            
            # Calculate accuracy - use model_device instead of global device
            acc = (logits.argmax(dim=-1) == labels.to(model_device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

        scheduler.step()
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        
        print(f"[ Train | Fold {fold_idx} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        model.eval()
        valid_loss = []
        valid_accs = []

        for batch in tqdm(valid_loader):
            imgs, labels = batch

            # Forward pass with TTA - use model_device
            with torch.no_grad():
                valid_pred = model(imgs.to(model_device))
            
            # TTA - use model_device
            tta_preds = []
            for _ in range(3):
                with torch.no_grad():
                    aug_imgs = train_tfm(imgs).to(model_device)
                    pred = model(aug_imgs)
                    tta_preds.append(pred)
            
            # Integrate prediction results
            avg_tta_pred = torch.mean(torch.stack(tta_preds), dim=0)
            final_pred = avg_tta_pred * 0.2 + valid_pred * 0.8
            
            # Other calculations - use model_device 
            loss = criterion(final_pred, labels.to(model_device))
            acc = (final_pred.argmax(dim=-1) == labels.to(model_device)).float().mean()
            
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Record to TensorBoard
        train_writer.add_scalar('Loss', train_loss, epoch)
        valid_writer.add_scalar('Loss', valid_loss, epoch)
        train_writer.add_scalar('Accuracy', train_acc, epoch)
        valid_writer.add_scalar('Accuracy', valid_acc, epoch)
        train_writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"[ Valid | Fold {fold_idx} | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # Save the best model
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{fold_exp_name}_best.ckpt")
            best_acc = valid_acc
            stale = 0


    train_writer.close()
    valid_writer.close()
    
    return best_acc, f"models/{fold_exp_name}_best.ckpt"

# Function to prepare data and model for training a single fold
def fold_init(fold_info, args):
    fold_idx = fold_info['fold_idx']
    train_fold_files = fold_info['train_files']
    valid_fold_files = fold_info['valid_files']
    
    print(f"\n{'='*20} Starting Fold {fold_idx}/{args.folds} {'='*20}")
    
    # Create datasets for this fold
    train_fold_set = FoodDataset(".", tfm=train_tfm, files=train_fold_files)
    valid_fold_set = FoodDataset(".", tfm=test_tfm, files=valid_fold_files)
    
    # Create data loaders
    train_loader = DataLoader(
        train_fold_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers // args.threads,  # Divide workers among threads
        pin_memory=True,
        persistent_workers=True
    )
    valid_loader = DataLoader(
        valid_fold_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2, 
        pin_memory=True,
        persistent_workers=True
    )
    
    # Create a new model for each fold to avoid interference
    fold_model, fold_exp_name = get_model(args.model)  # Get model name locally
    fold_model = fold_model.to(f"cuda:{fold_idx % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    
    # Train the fold - Pass fold_exp_name to train_fold
    best_acc, model_path = train_fold(fold_model, train_loader, valid_loader, fold_idx, args, fold_exp_name)
    print(f"Fold {fold_idx} best validation accuracy: {best_acc:.5f}")
    
    return best_acc, model_path, fold_idx

def cross_validation(args):
    global _exp_name, model
    n_folds = args.folds
    
    # Load all training data
    print("Loading all training data...")
    train_set = FoodDataset("./data/train", tfm=train_tfm)
    valid_set = FoodDataset("./data/valid", tfm=test_tfm)
    
    # Get all labels for stratified sampling
    all_labels = []
    all_files = []
    
    # When collecting training dataset files and labels
    for i, file_path in enumerate(train_set.files):
        try:
            # Extract label from filename
            label = int(file_path.split("/")[-1].split("_")[0])
            all_labels.append(label)
            all_files.append(file_path)
        except:
            continue  # Skip files without proper labels
    
    # Collect labels and files for validation set
    for i, file_path in enumerate(valid_set.files):
        try:
            # Extract label from filename
            label = int(file_path.split("/")[-1].split("_")[0])
            all_labels.append(label)
            all_files.append(file_path)
        except:
            continue  # Skip files without proper labels
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
    
    # Create fold data
    fold_data = []
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(all_files, all_labels)):
        train_fold_files = [all_files[i] for i in train_idx]
        valid_fold_files = [all_files[i] for i in valid_idx]
        
        fold_data.append({
            'fold_idx': fold_idx + 1,
            'train_files': train_fold_files,
            'valid_files': valid_fold_files
        })
    
    
    
    # Run folds in parallel using ThreadPoolExecutor
    fold_results = []
    
    if args.threads > 1:
        print(f"Training {n_folds} folds using {args.threads} parallel threads")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
            # Submit all fold training tasks
            future_to_fold = {
                executor.submit(fold_init, fold_info, args): fold_info['fold_idx'] 
                for fold_info in fold_data
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_fold):
                fold_idx = future_to_fold[future]
                try:
                    best_acc, model_path, _ = future.result()
                    fold_results.append((best_acc, model_path))
                    print(f"Fold {fold_idx} completed with accuracy: {best_acc:.5f}")
                except Exception as e:
                    print(f"Fold {fold_idx} generated an exception: {e}")
    else:
        print("Training folds sequentially")
        for fold_info in fold_data:
            best_acc, model_path, fold_idx = fold_init(fold_info, args)
            fold_results.append((best_acc, model_path))
    
    # Print cross-validation results
    print("\n" + "="*50)
    print("Cross-Validation Results:")
    mean_acc = 0
    for i, (acc, path) in enumerate(fold_results):
        print(f"Fold {i+1}: Accuracy = {acc:.5f}, Model = {path}")
        mean_acc += acc
    
    mean_acc /= n_folds
    print(f"Mean CV Accuracy: {mean_acc:.5f}")
    print("="*50)
    
    # Find the best model
    best_fold_idx = np.argmax([acc for acc, _ in fold_results])
    best_model_path = fold_results[best_fold_idx][1]
    print(f"Best model is from fold {best_fold_idx+1} with accuracy {fold_results[best_fold_idx][0]:.5f}")
    
    # Create final model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_model, _ = get_model(args.model)
    final_model = final_model.to(device)
    
    # Copy the best model as the final model
    final_model.load_state_dict(torch.load(best_model_path))
    torch.save(final_model.state_dict(), f"models/{_exp_name}_best.ckpt")
    
    return final_model

def ensemble_models(args):
    """
    Implements model ensembling using voting methods (hard or soft voting)
    
    Args:
        args: Arguments containing ensemble settings
        
    Returns:
        A final prediction through ensembling
    """
    # Load all test data
    print("Loading test data for ensemble prediction...")
    test_set = FoodDataset("./data/test", tfm=test_tfm)
    test_loader = DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Get all model checkpoints from the models directory
    if args.ensemble_models:
        # If specific models are provided, use them
        model_paths = args.ensemble_models.split(',')
    else:
        # Otherwise, use all the fold models for the current experiment
        model_paths = glob.glob(f"models/default/fold*_best.ckpt")
    
    print(f"Ensembling {len(model_paths)} models using {args.voting} voting")
    print("Models to ensemble:", model_paths)
    
    # Load all models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    
    for path in model_paths:
        # Determine model type from filename (you might need to adjust this based on your naming convention)
        if "resnet18" in path:
            model_type = "resnet"
        elif "mobilenet_v2" in path:
            model_type = "mobilenet" 
        elif "efficientnet_b0" in path:
            model_type = "efficientnet"
        else:
            model_type = "default"
        
        # Create model and load weights
        model, _ = get_model(model_type)
        model = model.to(device)
        model.load_state_dict(torch.load(path))
        model.eval()
        models.append(model)
    
    # Perform ensemble prediction
    num_models = len(models)
    num_classes = 11  # For Food-11 dataset
    all_predictions = []
    
    # Process each batch
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Initialize tensor to store all model predictions for this batch
            batch_probs = torch.zeros(num_models, batch_size, num_classes).to(device)
            
            # Get predictions from each model
            for model_idx, model in enumerate(models):
                # Basic prediction
                outputs = model(images)
                
                # TTA - Test Time Augmentation
                tta_preds = []
                for _ in range(3):
                    aug_imgs = train_tfm(images).to(device)
                    pred = model(aug_imgs)
                    tta_preds.append(pred)
                
                # Combine TTA with regular prediction
                avg_tta_pred = torch.mean(torch.stack(tta_preds), dim=0)
                final_pred = avg_tta_pred * 0.2 + outputs * 0.8
                
                # Convert logits to probabilities
                probs = torch.softmax(final_pred, dim=1)
                batch_probs[model_idx] = probs
            
            # Process predictions based on voting strategy
            if args.voting == 'hard':
                # Hard voting: each model votes for a class
                votes = batch_probs.argmax(dim=2)  # Shape: (num_models, batch_size)
                
                # Count votes for each class for each sample
                batch_predictions = []
                for sample_idx in range(batch_size):
                    sample_votes = votes[:, sample_idx]
                    
                    # Count votes for each class
                    vote_counts = {}
                    for vote in sample_votes:
                        vote = vote.item()
                        if vote not in vote_counts:
                            vote_counts[vote] = 0
                        vote_counts[vote] += 1
                    
                    # Find class with most votes
                    max_votes = 0
                    final_prediction = 0
                    for cls, count in vote_counts.items():
                        if count > max_votes:
                            max_votes = count
                            final_prediction = cls
                    
                    batch_predictions.append(final_prediction)
                
                all_predictions.extend(batch_predictions)
            
            else:  # Soft voting
                # Average probabilities across models
                avg_probs = torch.mean(batch_probs, dim=0)  # Shape: (batch_size, num_classes)
                batch_predictions = avg_probs.argmax(dim=1).cpu().numpy().tolist()
                all_predictions.extend(batch_predictions)
    
    # Create submission file
    def pad4(i):
        return "0" * (4 - len(str(i))) + str(i)
    
    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(len(test_set))]
    df["Category"] = all_predictions
    
    # Generate a unique name for the ensemble submission
    if args.ensemble_name:
        submission_name = f"submission_{args.ensemble_name}.csv"
    else:
        submission_name = f"submission_ensemble_{args.voting}_voting.csv"
    
    df.to_csv(submission_name, index=False)
    print(f"Ensemble predictions saved to {submission_name}")
    
    return all_predictions