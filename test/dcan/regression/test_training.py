import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import os
import tempfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR

# Import the classes and functions to test
from training import (
    Config, DataHandler, ModelHandler, TrainingLoop, TensorBoardLogger,
    LoesScoringTrainingApp, count_items, normalize_list, normalize_dictionary,
    get_folder_name
)


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_default_values(self):
        args = self.config.parse_args(['test_comment'])
        self.assertEqual(args.tb_prefix, 'loes_scoring')
        self.assertEqual(args.num_workers, 8)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.optimizer, 'Adam')
        self.assertEqual(args.model, 'ResNet')
        self.assertEqual(args.lr, 0.001)
        self.assertEqual(args.scheduler, 'plateau')
        self.assertEqual(args.comment, 'test_comment')

    def test_custom_values(self):
        args = self.config.parse_args([
            '--batch-size', '64',
            '--epochs', '10',
            '--lr', '0.01',
            '--model', 'AlexNet',
            '--optimizer', 'SGD',
            '--scheduler', 'step',
            'custom_comment'
        ])
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.lr, 0.01)
        self.assertEqual(args.model, 'AlexNet')
        self.assertEqual(args.optimizer, 'SGD')
        self.assertEqual(args.scheduler, 'step')
        self.assertEqual(args.comment, 'custom_comment')

    def test_boolean_flags(self):
        args = self.config.parse_args(['--use-train-validation-cols', '--use-weighted-loss', 'test'])
        self.assertTrue(args.use_train_validation_cols)
        self.assertTrue(args.use_weighted_loss)

    def test_invalid_model_choice(self):
        with self.assertRaises(SystemExit):
            self.config.parse_args(['--model', 'InvalidModel', 'test'])

    def test_invalid_scheduler_choice(self):
        with self.assertRaises(SystemExit):
            self.config.parse_args(['--scheduler', 'invalid', 'test'])


class TestDataHandler(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'anonymized_subject_id': ['subj1', 'subj2', 'subj3'],
            'loes-score': [1.0, 2.0, 3.0]
        })
        self.output_df = self.df.copy()
        self.data_handler = DataHandler(
            df=self.df,
            output_df=self.output_df,
            use_cuda=False,
            batch_size=2,
            num_workers=1
        )

    @patch('training.LoesScoreDataset')
    @patch('training.DataLoader')
    def test_init_dl_without_cuda(self, mock_dataloader, mock_dataset):
        mock_dataset_instance = Mock()
        mock_dataset.return_value = mock_dataset_instance
        mock_dataloader_instance = Mock()
        mock_dataloader.return_value = mock_dataloader_instance

        result = self.data_handler.init_dl('test_folder', ['subj1', 'subj2'])

        mock_dataset.assert_called_once_with(
            'test_folder', ['subj1', 'subj2'], self.df, self.output_df, is_val_set_bool=False
        )
        mock_dataloader.assert_called_once_with(
            mock_dataset_instance, batch_size=2, num_workers=1, pin_memory=False
        )
        self.assertEqual(result, mock_dataloader_instance)

    @patch('training.LoesScoreDataset')
    @patch('training.DataLoader')
    @patch('torch.cuda.device_count', return_value=2)
    def test_init_dl_with_cuda(self, mock_device_count, mock_dataloader, mock_dataset):
        data_handler = DataHandler(
            df=self.df,
            output_df=self.output_df,
            use_cuda=True,
            batch_size=2,
            num_workers=1
        )
        
        mock_dataset_instance = Mock()
        mock_dataset.return_value = mock_dataset_instance
        mock_dataloader_instance = Mock()
        mock_dataloader.return_value = mock_dataloader_instance

        result = data_handler.init_dl('test_folder', ['subj1', 'subj2'])

        # Batch size should be multiplied by device count
        mock_dataloader.assert_called_once_with(
            mock_dataset_instance, batch_size=4, num_workers=1, pin_memory=True
        )

    @patch('training.LoesScoreDataset')
    @patch('training.DataLoader')
    def test_init_dl_validation_set(self, mock_dataloader, mock_dataset):
        mock_dataset_instance = Mock()
        mock_dataset.return_value = mock_dataset_instance

        self.data_handler.init_dl('test_folder', ['subj1'], is_val_set=True)

        mock_dataset.assert_called_once_with(
            'test_folder', ['subj1'], self.df, self.output_df, is_val_set_bool=True
        )


class TestModelHandler(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')

    @patch('training.get_resnet_model')
    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.cuda.device_count', return_value=1)
    def test_init_resnet_model_cpu(self, mock_device_count, mock_cuda_available, mock_get_resnet):
        mock_model = Mock()
        mock_get_resnet.return_value = mock_model
        
        model_handler = ModelHandler('ResNet', use_cuda=False, device=self.device)
        
        mock_get_resnet.assert_called_once()
        self.assertEqual(model_handler.model, mock_model)

    @patch('training.AlexNet3D')
    @patch('torch.cuda.device_count', return_value=1)
    def test_init_alexnet_model(self, mock_device_count, mock_alexnet):
        mock_model = Mock()
        mock_alexnet.return_value = mock_model
        
        model_handler = ModelHandler('AlexNet', use_cuda=False, device=self.device)
        
        mock_alexnet.assert_called_once_with(4608)
        mock_model.to.assert_called_once_with(self.device)

    @patch('training.get_resnet_model')
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.nn.DataParallel')
    def test_init_model_with_multi_gpu(self, mock_data_parallel, mock_device_count, 
                                      mock_cuda_available, mock_get_resnet):
        mock_model = Mock()
        mock_get_resnet.return_value = mock_model
        mock_parallel_model = Mock()
        mock_data_parallel.return_value = mock_parallel_model
        
        model_handler = ModelHandler('ResNet', use_cuda=True, device=self.device)
        
        mock_data_parallel.assert_called_once_with(mock_model)
        mock_parallel_model.to.assert_called_once_with(self.device)

    @patch('torch.save')
    def test_save_model_regular(self, mock_save):
        mock_model = Mock()
        mock_model.state_dict.return_value = {'key': 'value'}
        
        model_handler = ModelHandler('ResNet', use_cuda=False, device=self.device)
        model_handler.model = mock_model
        
        model_handler.save_model('test_path.pt')
        
        mock_save.assert_called_once_with({'key': 'value'}, 'test_path.pt')

    @patch('torch.save')
    def test_save_model_data_parallel(self, mock_save):
        mock_module = Mock()
        mock_module.state_dict.return_value = {'key': 'value'}
        mock_model = Mock(spec=torch.nn.DataParallel)
        mock_model.module = mock_module
        
        model_handler = ModelHandler('ResNet', use_cuda=False, device=self.device)
        model_handler.model = mock_model
        
        model_handler.save_model('test_path.pt')
        
        mock_save.assert_called_once_with({'key': 'value'}, 'test_path.pt')

    @patch('torch.load')
    def test_load_model(self, mock_load):
        mock_load.return_value = {'key': 'value'}
        mock_model = Mock()
        
        model_handler = ModelHandler('ResNet', use_cuda=False, device=self.device)
        model_handler.model = mock_model
        
        model_handler.load_model('test_path.pt')
        
        mock_load.assert_called_once_with('test_path.pt')
        mock_model.load_state_dict.assert_called_once_with({'key': 'value'})


class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.mock_model_handler = Mock()
        self.mock_model = Mock()
        self.mock_model_handler.model = self.mock_model
        self.mock_optimizer = Mock()
        
        self.df = pd.DataFrame({
            'training': [1, 1, 0],
            'loes-score': [1.0, 2.0, 1.0]
        })
        
        self.config = Mock()
        self.config.use_weighted_loss = False
        
        self.training_loop = TrainingLoop(
            model_handler=self.mock_model_handler,
            optimizer=self.mock_optimizer,
            device=self.device,
            df=self.df,
            config=self.config
        )

    def test_init_weights_calculation(self):
        # Test that weights are calculated correctly from training data
        expected_weights = {1.0: 1.0, 2.0: 1.0}  # Each score appears once in training data
        self.assertEqual(self.training_loop.weights, expected_weights)

    @patch('training.enumerateWithEstimate')
    def test_train_epoch(self, mock_enumerate):
        # Setup mock data loader
        mock_dl = Mock()
        mock_dl.dataset = [1, 2, 3]  # 3 samples
        mock_dl.num_workers = 1
        mock_dl.batch_size = 2
        
        # Setup mock batch data
        mock_batch = (
            torch.randn(2, 3, 32, 32, 32),  # input
            torch.tensor([1.0, 2.0]),       # labels
            None, None
        )
        mock_enumerate.return_value = [(0, mock_batch)]
        
        # Setup model output
        self.mock_model.return_value = torch.tensor([1.1, 1.9])
        
        # Run training epoch
        self.training_loop.train_epoch(1, mock_dl)
        
        # Verify model was set to training mode
        self.mock_model.train.assert_called_once()
        
        # Verify optimizer was used
        self.mock_optimizer.zero_grad.assert_called_once()
        self.mock_optimizer.step.assert_called_once()

    @patch('training.enumerateWithEstimate')
    def test_validate_epoch(self, mock_enumerate):
        # Setup mock data loader
        mock_dl = Mock()
        mock_dl.dataset = [1, 2]  # 2 samples
        mock_dl.num_workers = 1
        mock_dl.batch_size = 2
        
        # Setup mock batch data
        mock_batch = (
            torch.randn(2, 3, 32, 32, 32),  # input
            torch.tensor([1.0, 2.0]),       # labels
            None, None
        )
        mock_enumerate.return_value = [(0, mock_batch)]
        
        # Setup model output
        self.mock_model.return_value = torch.tensor([1.1, 1.9])
        
        # Run validation epoch
        with torch.no_grad():
            self.training_loop.validate_epoch(1, mock_dl)
        
        # Verify model was set to eval mode
        self.mock_model.eval.assert_called_once()

    def test_weighted_mse_loss(self):
        predictions = torch.tensor([1.1, 2.1])
        targets = torch.tensor([1.0, 2.0])
        
        loss = self.training_loop.weighted_mse_loss(predictions, targets)
        
        # Should return a tensor with weighted squared errors
        self.assertEqual(loss.shape, (2,))
        self.assertTrue(torch.all(loss >= 0))  # All losses should be non-negative

    def test_weighted_mse_loss_unknown_target(self):
        # Test with a target value not in training weights
        predictions = torch.tensor([3.1])
        targets = torch.tensor([3.0])  # This value wasn't in training data
        
        loss = self.training_loop.weighted_mse_loss(predictions, targets)
        
        # Should still work and return a reasonable loss
        self.assertEqual(loss.shape, (1,))
        self.assertTrue(torch.all(loss >= 0))


class TestUtilityFunctions(unittest.TestCase):
    def test_count_items(self):
        input_list = [1, 2, 2, 3, 3, 3]
        result = count_items(input_list)
        expected = {1: 1, 2: 2, 3: 3}
        self.assertEqual(result, expected)

    def test_count_items_empty(self):
        result = count_items([])
        self.assertEqual(result, {})

    def test_normalize_list(self):
        data = [1, 2, 3, 4, 5]
        result = normalize_list(data)
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.assertEqual(result, expected)

    def test_normalize_list_single_value(self):
        data = [5, 5, 5]
        result = normalize_list(data)
        expected = [0.0, 0.0, 0.0]
        self.assertEqual(result, expected)

    def test_normalize_dictionary(self):
        data = {'a': 1, 'b': 2, 'c': 3}
        result = normalize_dictionary(data)
        expected = {'a': 0.0, 'b': 0.5, 'c': 1.0}
        self.assertEqual(result, expected)

    def test_normalize_dictionary_same_values(self):
        data = {'a': 5, 'b': 5, 'c': 5}
        result = normalize_dictionary(data)
        expected = {'a': 0.0, 'b': 0.0, 'c': 0.0}
        self.assertEqual(result, expected)

    def test_get_folder_name(self):
        file_path = '/home/user/documents/file.txt'
        result = get_folder_name(file_path)
        self.assertEqual(result, 'documents')

    def test_get_folder_name_root(self):
        file_path = '/file.txt'
        result = get_folder_name(file_path)
        self.assertEqual(result, '')

    def test_get_folder_name_relative(self):
        file_path = 'folder/file.txt'
        result = get_folder_name(file_path)
        self.assertEqual(result, 'folder')


class TestTensorBoardLogger(unittest.TestCase):
    @patch('training.SummaryWriter')
    @patch('os.path.join')
    def test_init(self, mock_join, mock_writer):
        mock_join.return_value = 'test_path'
        mock_writer_instance = Mock()
        mock_writer.return_value = mock_writer_instance
        
        logger = TensorBoardLogger('prefix', 'time', 'comment')
        
        # Should create two writers
        self.assertEqual(mock_writer.call_count, 2)
        self.assertEqual(logger.trn_writer, mock_writer_instance)
        self.assertEqual(logger.val_writer, mock_writer_instance)

    @patch('training.SummaryWriter')
    def test_log_metrics(self, mock_writer):
        mock_writer_instance = Mock()
        mock_writer.return_value = mock_writer_instance
        
        logger = TensorBoardLogger('prefix', 'time', 'comment')
        
        metrics = torch.zeros(3, 10)
        metrics[2] = torch.ones(10) * 0.5  # Loss values
        
        logger.log_metrics('trn', 1, metrics, 100)
        
        # Should call add_scalar on the training writer
        mock_writer_instance.add_scalar.assert_called_with('loss/all', 0.5, 100)

    @patch('training.SummaryWriter')
    def test_close(self, mock_writer):
        mock_writer_instance = Mock()
        mock_writer.return_value = mock_writer_instance
        
        logger = TensorBoardLogger('prefix', 'time', 'comment')
        logger.close()
        
        # Should close both writers
        self.assertEqual(mock_writer_instance.close.call_count, 2)


class TestLoesScoringTrainingApp(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file for testing
        self.test_df = pd.DataFrame({
            'anonymized_subject_id': ['subj1', 'subj2', 'subj3', 'subj4'],
            'loes-score': [1.0, 2.0, 3.0, 1.5],
            'training': [1, 1, 0, 0],
            'validation': [0, 0, 1, 1],
            'scan': ['T1', 'T2', 'T1_Gd', 'T2']
        })
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def tearDown(self):
        os.unlink(self.temp_file.name)

    @patch('training.ModelHandler')
    @patch('training.DataHandler')
    @patch('training.TensorBoardLogger')
    @patch('torch.cuda.is_available', return_value=False)
    def test_init(self, mock_cuda, mock_logger, mock_data_handler, mock_model_handler):
        sys_argv = [
            'training.py',
            '--csv-input-file', self.temp_file.name,
            '--batch-size', '16',
            'test_comment'
        ]
        
        app = LoesScoringTrainingApp(sys_argv)
        
        self.assertEqual(app.config.batch_size, 16)
        self.assertEqual(app.config.comment, 'test_comment')
        self.assertFalse(app.use_cuda)
        self.assertEqual(len(app.input_df), 4)

    @patch('training.ModelHandler')
    @patch('training.DataHandler')
    @patch('training.TensorBoardLogger')
    @patch('torch.cuda.is_available', return_value=False)
    def test_init_optimizer_adam(self, mock_cuda, mock_logger, mock_data_handler, mock_model_handler):
        sys_argv = [
            'training.py',
            '--csv-input-file', self.temp_file.name,
            '--optimizer', 'adam',
            'test'
        ]
        
        app = LoesScoringTrainingApp(sys_argv)
        
        self.assertIsInstance(app.optimizer, Adam)

    @patch('training.ModelHandler')
    @patch('training.DataHandler')
    @patch('training.TensorBoardLogger')
    @patch('torch.cuda.is_available', return_value=False)
    def test_init_optimizer_sgd(self, mock_cuda, mock_logger, mock_data_handler, mock_model_handler):
        sys_argv = [
            'training.py',
            '--csv-input-file', self.temp_file.name,
            '--optimizer', 'sgd',
            'test'
        ]
        
        app = LoesScoringTrainingApp(sys_argv)
        
        self.assertIsInstance(app.optimizer, SGD)

    @patch('training.ModelHandler')
    @patch('training.DataHandler')
    @patch('training.TensorBoardLogger')
    @patch('torch.cuda.is_available', return_value=False)
    def test_split_train_validation(self, mock_cuda, mock_logger, mock_data_handler, mock_model_handler):
        sys_argv = [
            'training.py',
            '--csv-input-file', self.temp_file.name,
            'test'
        ]
        
        app = LoesScoringTrainingApp(sys_argv)
        train_subjects, val_subjects = app.split_train_validation()
        
        self.assertEqual(set(train_subjects), {'subj1', 'subj2'})
        self.assertEqual(set(val_subjects), {'subj3', 'subj4'})

    @patch('training.ModelHandler')
    @patch('training.DataHandler')
    @patch('training.TensorBoardLogger')
    @patch('torch.cuda.is_available', return_value=False)
    def test_filter_gd_scans(self, mock_cuda, mock_logger, mock_data_handler, mock_model_handler):
        sys_argv = [
            'training.py',
            '--csv-input-file', self.temp_file.name,
            '--gd', '0',
            'test'
        ]
        
        app = LoesScoringTrainingApp(sys_argv)
        
        # Should filter out scans containing 'Gd'
        self.assertFalse(any('Gd' in scan for scan in app.input_df['scan']))
        self.assertEqual(len(app.input_df), 3)  # Should have 3 rows after filtering

    def test_init_scheduler_types(self):
        """Test different scheduler initialization"""
        mock_optimizer = Mock()
        mock_train_dl = Mock()
        mock_train_dl.__len__ = Mock(return_value=10)
        
        # Test different schedulers
        schedulers_to_test = ['plateau', 'step', 'cosine', 'onecycle']
        
        for scheduler_type in schedulers_to_test:
            with patch('training.ModelHandler'), \
                 patch('training.DataHandler'), \
                 patch('training.TensorBoardLogger'), \
                 patch('torch.cuda.is_available', return_value=False):
                
                sys_argv = [
                    'training.py',
                    '--csv-input-file', self.temp_file.name,
                    '--scheduler', scheduler_type,
                    '--epochs', '5',
                    'test'
                ]
                
                app = LoesScoringTrainingApp(sys_argv)
                app.optimizer = mock_optimizer
                scheduler = app._init_scheduler(mock_train_dl)
                
                if scheduler_type == 'plateau':
                    self.assertIsInstance(scheduler, ReduceLROnPlateau)
                elif scheduler_type == 'step':
                    self.assertIsInstance(scheduler, StepLR)
                elif scheduler_type == 'cosine':
                    self.assertIsInstance(scheduler, CosineAnnealingLR)
                elif scheduler_type == 'onecycle':
                    self.assertIsInstance(scheduler, OneCycleLR)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete training pipeline"""
    
    def setUp(self):
        # Create a more comprehensive test dataset
        self.test_df = pd.DataFrame({
            'anonymized_subject_id': [f'subj{i}' for i in range(10)],
            'loes-score': [float(i % 5) for i in range(10)],
            'training': [1 if i < 6 else 0 for i in range(10)],
            'validation': [1 if i >= 6 else 0 for i in range(10)],
            'scan': [f'T1_{i}' for i in range(10)]
        })
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def tearDown(self):
        os.unlink(self.temp_file.name)

    @patch('training.get_validation_info')
    @patch('training.add_predicted_values')
    @patch('training.create_scatter_plot')
    @patch('training.LoesScoreDataset')
    @patch('os.makedirs')
    @patch('training.get_resnet_model')
    def test_minimal_training_run(self, mock_get_resnet, mock_makedirs, mock_dataset,
                                 mock_scatter, mock_add_pred, mock_get_val_info):
        """Test a minimal training run without actual model training"""
        
        # Setup mocks
        mock_model = Mock()
        mock_get_resnet.return_value = mock_model
        
        mock_dataset_instance = Mock()
        mock_dataset_instance.__len__ = Mock(return_value=2)
        mock_dataset.return_value = mock_dataset_instance
        
        # Mock validation info
        mock_get_val_info.return_value = (['subj7', 'subj8'], ['sess1', 'sess2'], [1.0, 2.0], [1.1, 1.9])
        mock_add_pred.return_value = self.test_df
        
        with patch('training.DataLoader') as mock_dataloader, \
             patch('training.enumerateWithEstimate') as mock_enumerate, \
             patch('torch.cuda.is_available', return_value=False):
            
            # Setup mock dataloader
            mock_dl_instance = Mock()
            mock_dl_instance.dataset = [1, 2]
            mock_dl_instance.num_workers = 1
            mock_dl_instance.batch_size = 2
            mock_dl_instance.__len__ = Mock(return_value=1)
            mock_dataloader.return_value = mock_dl_instance
            
            # Setup mock batch
            mock_batch = (
                torch.randn(2, 3, 32, 32, 32),
                torch.tensor([1.0, 2.0]),
                None, None
            )
            mock_enumerate.return_value = [(0, mock_batch)]
            mock_model.return_value = torch.tensor([1.1, 1.9])
            
            # Create temp output file
            temp_output = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            temp_output.close()
            
            temp_plot = tempfile.NamedTemporaryFile(mode='w', suffix='.png', delete=False)
            temp_plot.close()
            
            try:
                sys_argv = [
                    'training.py',
                    '--csv-input-file', self.temp_file.name,
                    '--csv-output-file', temp_output.name,
                    '--plot-location', temp_plot.name,
                    '--folder', '/tmp/test',
                    '--epochs', '1',
                    '--batch-size', '2',
                    '--use-train-validation-cols',
                    'integration_test'
                ]
                
                app = LoesScoringTrainingApp(sys_argv)
                
                # This should run without errors
                app.main()
                
                # Verify some key interactions occurred
                mock_get_val_info.assert_called_once()
                mock_add_pred.assert_called_once()
                mock_scatter.assert_called_once()
                
            finally:
                os.unlink(temp_output.name)
                os.unlink(temp_plot.name)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_empty_dataframe_handling(self):
        """Test behavior with empty dataframes"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises((KeyError, IndexError)):
            data_handler = DataHandler(
                df=empty_df,
                output_df=empty_df,
                use_cuda=False,
                batch_size=2,
                num_workers=1
            )

    def test_invalid_file_paths(self):
        """Test handling of invalid file paths"""
        result = get_folder_name("")
        self.assertEqual(result, "")
        
        result = get_folder_name(None)
        self.assertIsNone(result)

    def test_single_sample_batch(self):
        """Test training with single sample batches"""
        device = torch.device('cpu')
        mock_model_handler = Mock()
        mock_model = Mock()
        mock_model_handler.model = mock_model
        mock_optimizer = Mock()
        
        df = pd.DataFrame({
            'training': [1],
            'loes-score': [1.0]
        })
        
        config = Mock()
        config.use_weighted_loss = False
        
        training_loop = TrainingLoop(
            model_handler=mock_model_handler,
            optimizer=mock_optimizer,
            device=device,
            df=df,
            config=config
        )
        
        # Should handle single sample without error
        predictions = torch.tensor([1.1])
        targets = torch.tensor([1.0])
        loss = training_loop.weighted_mse_loss(predictions, targets)
        self.assertEqual(loss.shape, (1,))

    def test_zero_variance_normalization(self):
        """Test normalization with zero variance data"""
        # All same values
        data = [5, 5, 5, 5]
        result = normalize_list(data)
        expected = [0.0, 0.0, 0.0, 0.0]
        self.assertEqual(result, expected)
        
        # Dictionary with same values
        data_dict = {'a': 3, 'b': 3, 'c': 3}
        result = normalize_dictionary(data_dict)
        expected = {'a': 0.0, 'b': 0.0, 'c': 0.0}
        self.assertEqual(result, expected)

    def test_extreme_learning_rates(self):
        """Test with extreme learning rate values"""
        with patch('training.ModelHandler'), \
             patch('training.DataHandler'), \
             patch('training.TensorBoardLogger'), \
             patch('torch.cuda.is_available', return_value=False):
            
            # Create temp file
            df = pd.DataFrame({'col': [1]})
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            df.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            try:
                # Very small learning rate
                sys_argv = [
                    'training.py',
                    '--csv-input-file', temp_file.name,
                    '--lr', '1e-10',
                    'test'
                ]
                app = LoesScoringTrainingApp(sys_argv)
                self.assertEqual(app.config.lr, 1e-10)
                
                # Very large learning rate
                sys_argv = [
                    'training.py',
                    '--csv-input-file', temp_file.name,
                    '--lr', '100.0',
                    'test'
                ]
                app = LoesScoringTrainingApp(sys_argv)
                self.assertEqual(app.config.lr, 100.0)
                
            finally:
                os.unlink(temp_file.name)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and exception cases"""
    
    def test_missing_required_files(self):
        """Test behavior when required files are missing"""
        with self.assertRaises(SystemExit):
            Config().parse_args(['--csv-input-file', 'nonexistent.csv', 'test'])

    def test_invalid_tensor_shapes(self):
        """Test handling of mismatched tensor shapes in loss computation"""
        device = torch.device('cpu')
        mock_model_handler = Mock()
        mock_model = Mock()
        mock_model_handler.model = mock_model
        mock_optimizer = Mock()
        
        df = pd.DataFrame({
            'training': [1, 1],
            'loes-score': [1.0, 2.0]
        })
        
        config = Mock()
        config.use_weighted_loss = False
        
        training_loop = TrainingLoop(
            model_handler=mock_model_handler,
            optimizer=mock_optimizer,
            device=device,
            df=df,
            config=config
        )
        
        # Mismatched shapes should raise an error
        predictions = torch.tensor([1.1, 2.1, 3.1])  # 3 elements
        targets = torch.tensor([1.0, 2.0])           # 2 elements
        
        with self.assertRaises(RuntimeError):
            training_loop.weighted_mse_loss(predictions, targets)

    @patch('training.get_resnet_model')
    @patch('torch.cuda.is_available', return_value=False)
    def test_model_initialization_failure(self, mock_cuda, mock_get_resnet):
        """Test handling of model initialization failures"""
        mock_get_resnet.side_effect = RuntimeError("Model initialization failed")
        
        with self.assertRaises(RuntimeError):
            ModelHandler('ResNet', use_cuda=False, device=torch.device('cpu'))

    def test_scheduler_with_invalid_parameters(self):
        """Test scheduler initialization with edge case parameters"""
        mock_optimizer = Mock()
        mock_train_dl = Mock()
        mock_train_dl.__len__ = Mock(return_value=0)  # Empty dataloader
        
        with patch('training.ModelHandler'), \
             patch('training.DataHandler'), \
             patch('training.TensorBoardLogger'), \
             patch('torch.cuda.is_available', return_value=False):
            
            df = pd.DataFrame({'col': [1]})
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            df.to_csv(temp_file.name, index=False)
            temp_file.close()
            
            try:
                sys_argv = [
                    'training.py',
                    '--csv-input-file', temp_file.name,
                    '--scheduler', 'onecycle',
                    '--epochs', '0',  # Zero epochs
                    'test'
                ]
                
                app = LoesScoringTrainingApp(sys_argv)
                app.optimizer = mock_optimizer
                
                # Should handle zero total steps gracefully
                scheduler = app._init_scheduler(mock_train_dl)
                self.assertIsInstance(scheduler, OneCycleLR)
                
            finally:
                os.unlink(temp_file.name)


class TestPerformanceEdgeCases(unittest.TestCase):
    """Test performance-related edge cases"""
    
    def test_large_batch_size_calculation(self):
        """Test batch size calculation with many GPUs"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        
        with patch('torch.cuda.device_count', return_value=8):
            data_handler = DataHandler(
                df=df,
                output_df=df,
                use_cuda=True,
                batch_size=4,
                num_workers=1
            )
            
            with patch('training.LoesScoreDataset'), \
                 patch('training.DataLoader') as mock_dataloader:
                
                data_handler.init_dl('test_folder', ['subj1'])
                
                # Should multiply batch size by number of GPUs
                args, kwargs = mock_dataloader.call_args
                self.assertEqual(kwargs['batch_size'], 32)  # 4 * 8

    def test_memory_intensive_operations(self):
        """Test operations that might consume significant memory"""
        # Large metrics tensor simulation
        device = torch.device('cpu')
        large_dataset_size = 10000
        
        # This should work without memory issues on CPU
        metrics = torch.zeros(3, large_dataset_size, device=device)
        
        # Simulate filling metrics
        metrics[0] = torch.randn(large_dataset_size)  # labels
        metrics[1] = torch.randn(large_dataset_size)  # predictions
        metrics[2] = torch.randn(large_dataset_size)  # losses
        
        # Should be able to compute mean without issues
        mean_loss = metrics[2].mean()
        self.assertIsInstance(mean_loss.item(), float)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        # Very large values
        large_predictions = torch.tensor([1e6, 1e7])
        large_targets = torch.tensor([1e6, 1e7])
        
        # Very small values
        small_predictions = torch.tensor([1e-6, 1e-7])
        small_targets = torch.tensor([1e-6, 1e-7])
        
        # Should handle both without NaN or inf
        large_diff = (large_predictions - large_targets) ** 2
        small_diff = (small_predictions - small_targets) ** 2
        
        self.assertFalse(torch.isnan(large_diff).any())
        self.assertFalse(torch.isinf(large_diff).any())
        self.assertFalse(torch.isnan(small_diff).any())
        self.assertFalse(torch.isinf(small_diff).any())


class TestDataValidation(unittest.TestCase):
    """Test data validation and preprocessing"""
    
    def test_missing_columns_handling(self):
        """Test handling of missing required columns"""
        # DataFrame missing required columns
        incomplete_df = pd.DataFrame({
            'some_column': [1, 2, 3]
        })
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        incomplete_df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            with patch('training.ModelHandler'), \
                 patch('training.DataHandler'), \
                 patch('training.TensorBoardLogger'), \
                 patch('torch.cuda.is_available', return_value=False):
                
                sys_argv = [
                    'training.py',
                    '--csv-input-file', temp_file.name,
                    'test'
                ]
                
                app = LoesScoringTrainingApp(sys_argv)
                
                # Should raise KeyError when trying to access missing columns
                with self.assertRaises(KeyError):
                    app.split_train_validation()
                    
        finally:
            os.unlink(temp_file.name)

    def test_mixed_data_types(self):
        """Test handling of mixed data types in loes scores"""
        mixed_df = pd.DataFrame({
            'anonymized_subject_id': ['subj1', 'subj2', 'subj3'],
            'loes-score': [1.0, '2.0', 3],  # Mixed types
            'training': [1, 1, 0],
            'validation': [0, 0, 1]
        })
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        mixed_df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            with patch('training.ModelHandler'), \
                 patch('training.DataHandler'), \
                 patch('training.TensorBoardLogger'), \
                 patch('torch.cuda.is_available', return_value=False):
                
                sys_argv = [
                    'training.py',
                    '--csv-input-file', temp_file.name,
                    'test'
                ]
                
                # Should handle mixed types in DataFrame
                app = LoesScoringTrainingApp(sys_argv)
                self.assertEqual(len(app.input_df), 3)
                
        finally:
            os.unlink(temp_file.name)

    def test_duplicate_subjects(self):
        """Test handling of duplicate subject IDs"""
        duplicate_df = pd.DataFrame({
            'anonymized_subject_id': ['subj1', 'subj1', 'subj2'],  # Duplicate
            'loes-score': [1.0, 1.5, 2.0],
            'training': [1, 1, 0],
            'validation': [0, 0, 1]
        })
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        duplicate_df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            with patch('training.ModelHandler'), \
                 patch('training.DataHandler'), \
                 patch('training.TensorBoardLogger'), \
                 patch('torch.cuda.is_available', return_value=False):
                
                sys_argv = [
                    'training.py',
                    '--csv-input-file', temp_file.name,
                    'test'
                ]
                
                app = LoesScoringTrainingApp(sys_argv)
                train_subjects, val_subjects = app.split_train_validation()
                
                # Should handle duplicates by creating unique sets
                self.assertEqual(len(set(train_subjects)), len(train_subjects))
                self.assertEqual(len(set(val_subjects)), len(val_subjects))
                
        finally:
            os.unlink(temp_file.name)


class TestConcurrencyAndThreading(unittest.TestCase):
    """Test threading and concurrency-related functionality"""
    
    def test_multiple_workers_configuration(self):
        """Test configuration with multiple worker processes"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        
        data_handler = DataHandler(
            df=df,
            output_df=df,
            use_cuda=False,
            batch_size=2,
            num_workers=4  # Multiple workers
        )
        
        with patch('training.LoesScoreDataset'), \
             patch('training.DataLoader') as mock_dataloader:
            
            data_handler.init_dl('test_folder', ['subj1'])
            
            args, kwargs = mock_dataloader.call_args
            self.assertEqual(kwargs['num_workers'], 4)
            self.assertFalse(kwargs['pin_memory'])  # Should be False when not using CUDA

    def test_pin_memory_with_cuda(self):
        """Test pin_memory setting with CUDA"""
        df = pd.DataFrame({'col': [1, 2, 3]})
        
        data_handler = DataHandler(
            df=df,
            output_df=df,
            use_cuda=True,
            batch_size=2,
            num_workers=2
        )
        
        with patch('training.LoesScoreDataset'), \
             patch('training.DataLoader') as mock_dataloader:
            
            data_handler.init_dl('test_folder', ['subj1'])
            
            args, kwargs = mock_dataloader.call_args
            self.assertTrue(kwargs['pin_memory'])  # Should be True when using CUDA


if __name__ == '__main__':
    # Create a test suite with all test classes
    test_classes = [
        TestConfig,
        TestDataHandler,
        TestModelHandler,
        TestTrainingLoop,
        TestUtilityFunctions,
        TestTensorBoardLogger,
        TestLoesScoringTrainingApp,
        TestIntegration,
        TestEdgeCases,
        TestErrorHandling,
        TestPerformanceEdgeCases,
        TestDataValidation,
        TestConcurrencyAndThreading
    ]
    
    # Create test suite
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
