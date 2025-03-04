import os
import argparse
from config.config import config, seed_everything
from train import train
from evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description='Skeleton Action Recognition')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'train_eval'],
                        help='Run mode (train, eval, or train_eval)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing the dataset')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Maximum number of epochs for training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Update config with command line arguments if provided
    if args.data_dir:
        config['DATA_DIR'] = args.data_dir
    if args.result_dir:
        config['RESULT_DIR'] = args.result_dir
    if args.batch_size:
        config['BATCH_SIZE'] = args.batch_size
    if args.epochs:
        config['MAX_EPOCHS'] = args.epochs
    if args.lr:
        config['INIT_LR'] = args.lr

    # Set seed if provided
    if args.seed:
        seed_everything(args.seed)
    else:
        seed_everything()

    # Create result directory if it doesn't exist
    os.makedirs(config['RESULT_DIR'], exist_ok=True)

    # Execute based on mode
    if args.mode == 'train' or args.mode == 'train_eval':
        print("=== Starting Training ===")
        model, test_loader = train()

    if args.mode == 'eval' or args.mode == 'train_eval':
        print("\n=== Starting Evaluation ===")
        metrics = evaluate_model()

        # Print summarized results
        print("\n=== Final Results ===")
        print(f"Top-1 Accuracy: {metrics['Top-1 Accuracy']:.2f}%")
        print(f"Top-5 Accuracy: {metrics['Top-5 Accuracy']:.2f}%")
        print(f"F1 Score: {metrics['F1 Score']:.4f}")


if __name__ == "__main__":
    main()