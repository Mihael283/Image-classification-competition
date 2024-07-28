import logging
import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, model_name, ids_to_classnames, log_dir='logs'):
        self.ids_to_classnames = ids_to_classnames
        model_name = model_name.replace(' ', '_').lower()
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create a unique filename based on the current timestamp
        timestamp = datetime.now().strftime("%d%m%Y_%H%M")
        self.log_file = os.path.join(log_dir, f'training_log_{model_name}_{timestamp}.log')
        self.csv_file = os.path.join(log_dir, f'training_data_{model_name}_{timestamp}.csv')

        # Set up logging
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Initialize CSV file with headers
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'])

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        # Log to file
        log_message = f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
        logging.info(log_message)

        # Save to CSV
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        # Print to console
        print(log_message)

    def log_best_model(self, epoch, val_loss):
        message = f'New best model saved at epoch {epoch} with validation accuracy: {val_loss:.4f}'
        logging.info(message)
        print(message)

    def log_class_accuracy(self, epoch, class_accuracy, problematic_classes):
        log_message = f"\nEpoch {epoch} - Per-class accuracy:\n"
        for i, acc in enumerate(class_accuracy):
            class_name = self.ids_to_classnames[i]
            log_message += f"{class_name} (Class {i}): {acc:.2f}\n"
        
        log_message += "Classes with accuracy below 80%:\n"
        for index, name in problematic_classes:
            log_message += f"{name} (Class {index}): {class_accuracy[index]:.2f}\n"
            
        #logging.info(log_message)
