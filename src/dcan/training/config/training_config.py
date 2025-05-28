import argparse
import datetime
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Data parameters
    csv_input_file: str
    csv_output_file: Optional[str] = None
    features: List[str] = None
    target: str = None
    folder: Optional[str] = None
    use_train_validation_cols: bool = False
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.01
    num_workers: int = 4
    split_ratio: float = 0.8
    optimizer: str = 'Adam'
    scheduler: str = 'plateau'
    weight_decay: float = 0.0001
    model_type: str = 'conv'
    augment_minority: bool = False
    num_augmentations: int = 3
    
    # Output and tracking
    tb_prefix: str = 'logistic_regression'
    model_save_location: Optional[str] = None
    plot_location: Optional[str] = None
    comment: str = ''
    normalize_features: bool = False
    threshold: float = 0.5
    DEBUG: bool = False
    
    # Computed properties
    threshold_optimization: bool = True
    pauc_fpr_limit: float = 0.1

class ConfigurationManager:
    """Handles argument parsing and configuration creation"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self):
        parser = argparse.ArgumentParser()
        
        # Add all arguments (move from your existing Config class)
        parser.add_argument('--DEBUG', action='store_true')
        parser.add_argument('--csv-input-file', required=True, help="CSV data file")
        parser.add_argument('--csv-output-file', help="CSV output file for predictions")
        parser.add_argument('--features', required=True, nargs='+', help="Feature column names")
        parser.add_argument('--target', required=True, help="Target column name")
        parser.add_argument('--folder', help='Folder where MRIs are stored')
        parser.add_argument('--use_train_validation_cols', action='store_true')
        
        # Training parameters
        parser.add_argument('--batch-size', default=32, type=int)
        parser.add_argument('--epochs', default=100, type=int)
        parser.add_argument('--lr', default=0.01, type=float)
        parser.add_argument('--num-workers', default=4, type=int)
        parser.add_argument('--split-ratio', default=0.8, type=float)
        parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
        parser.add_argument('--scheduler', default='plateau', 
                          choices=['plateau', 'step', 'cosine', 'onecycle'])
        parser.add_argument('--weight-decay', default=0.0001, type=float)
        parser.add_argument('--model-type', default='conv', 
                          choices=['conv', 'simple', 'resnet3d', 'dense3d', 'efficientnet3d'])
        parser.add_argument('--augment-minority', action='store_true')
        parser.add_argument('--num-augmentations', type=int, default=3)
        
        # Output and tracking
        parser.add_argument('--tb-prefix', default='logistic_regression')
        parser.add_argument('--model-save-location')
        parser.add_argument('--plot-location')
        parser.add_argument('--comment', default='')
        parser.add_argument('--normalize-features', action='store_true')
        parser.add_argument('--threshold', default=0.5, type=float)
        
        return parser
    
    def parse_config(self, sys_argv=None) -> TrainingConfig:
        args = self.parser.parse_args(sys_argv)
        
        # Set defaults
        if not args.model_save_location:
            args.model_save_location = f'./logistic_model-{datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")}.pt'
        
        if not args.plot_location and args.csv_output_file:
            args.plot_location = os.path.splitext(args.csv_output_file)[0] + "_plot.png"
        
        # Create config object
        config = TrainingConfig(
            csv_input_file=args.csv_input_file,
            csv_output_file=args.csv_output_file,
            features=args.features,
            target=args.target,
            folder=args.folder,
            use_train_validation_cols=args.use_train_validation_cols,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            num_workers=args.num_workers,
            split_ratio=args.split_ratio,
            optimizer=args.optimizer,
            scheduler=args.scheduler,
            weight_decay=args.weight_decay,
            model_type=args.model_type,
            augment_minority=args.augment_minority,
            num_augmentations=args.num_augmentations,
            tb_prefix=args.tb_prefix,
            model_save_location=args.model_save_location,
            plot_location=args.plot_location,
            comment=args.comment,
            normalize_features=args.normalize_features,
            threshold=args.threshold,
            DEBUG=args.DEBUG
        )
        
        self._validate_config(config)
        return config
    
    def _validate_config(self, config: TrainingConfig):
        """Validate configuration (move validation logic here)"""
        if not os.path.exists(config.csv_input_file):
            raise FileNotFoundError(f"Input CSV file not found: {config.csv_input_file}")
        
        if config.folder and not os.path.exists(config.folder):
            raise FileNotFoundError(f"MRI folder not found: {config.folder}")
        
        # TODO Add other validations...
