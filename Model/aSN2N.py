import os
import random
import torch
import datetime
import logging
from pathlib import Path
import numpy as np
from glob import glob
from skimage.io import imread, imsave
from torch.utils.tensorboard import SummaryWriter
from .AUnet import AUnet
from .loss import L0Loss
import matplotlib.pyplot as plt


class aSN2N():
    def __init__(self, dataset_name, tests_name, reg=1, reg_sparse=0,
                 constrained_type='L1', lr=2e-4, epochs=100, train_batch_size=32,
                 ifadaptive_lr=False, test_batch_size=1, img_res=(128, 128),
                 train_data_path=None, test_path=None, test_mode=False, work_mode='local', 
                 inference_mode='local',
                 # --- parameters for overlapping patch inference ---
                 inference_patch_size=None, # e.g., 256 or 512. If None, uses img_res
                 inference_stride=None,     # e.g., 128 or 256. If None, uses patch_size // 2
                 gaussian_sigma_scale=4):   # Sigma = patch_size / scale. Larger scale -> flatter Gaussian
        
        # Basic parameter initialization
        self.test_mode = test_mode  # Test mode
        self.work_mode = work_mode  # global/local normalization
        self.inference_mode = inference_mode  # Inference with overlapping patches
        self.dataset_name = dataset_name
        self.tests_name = tests_name
        self.reg = reg
        self.reg_sparse = reg_sparse
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AUnet(n_channels=1, n_classes=1, bilinear=True).to(self.device)
        self.img_res = img_res
        self.train_batch_size = train_batch_size
        self.epochs = epochs
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.constrained_type = constrained_type
        self.train_data_path = train_data_path
        self.test_path = test_path

        # --- Overlapping Patch Parameters ---
        self.inference_patch_size = inference_patch_size if inference_patch_size is not None else self.img_res[0]
        # Ensure patch size is even for easier division if needed, though not strictly required
        if self.inference_patch_size % 2 != 0:
             self.inference_patch_size +=1
             print(f"Adjusted inference_patch_size to be even: {self.inference_patch_size}")

        self.inference_stride = inference_stride if inference_stride is not None else self.inference_patch_size // 2
        self.gaussian_sigma = self.inference_patch_size / gaussian_sigma_scale
        self._gaussian_weights = self._generate_gaussian_weights(
            self.inference_patch_size, self.gaussian_sigma
        ).to(self.device) # Precompute Gaussian weights

        # Create checkpoint directory
        self.checkpoint_dir = Path(f'./images/{self.tests_name}/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._setup_logging()

        # Initialize TensorBoard
        self.writer = SummaryWriter(log_dir=f'./images/{self.tests_name}/runs/')
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Set up loss functions
        self._setup_loss_functions()

        # Learning rate scheduling
        self.ifadaptive_lr = ifadaptive_lr
        if self.ifadaptive_lr:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        # Initialize training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.train_history = []

        self.logger.info(f"Initialized aSN2N for {dataset_name}/{tests_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training image resolution: {self.img_res}")
        self.logger.info(f"Inference patch size: {self.inference_patch_size}, Stride: {self.inference_stride}, Gaussian Sigma: {self.gaussian_sigma:.2f}")
        self.logger.info(f"Normalization mode (training): {self.work_mode}")


    def _setup_logging(self):
        """Set up logging."""
        log_file = self.checkpoint_dir / f'training_{datetime.datetime.now():%Y%m%d_%H%M%S}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__) # Use class name for logger


    def _setup_loss_functions(self):
        """Set up loss functions."""
        if self.constrained_type == 'L1':
            self.constrained = torch.nn.L1Loss(reduction='mean')
            self.criterion = torch.nn.L1Loss(reduction='mean')
        elif self.constrained_type == 'L0':
            self.constrained = L0Loss()
            self.criterion = L0Loss()
        elif self.constrained_type == 'SmoothL1':
            self.constrained = torch.nn.SmoothL1Loss(reduction='mean')
            self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
        elif self.constrained_type == 'None':
            self.reg = 0
            self.criterion = torch.nn.L1Loss(reduction='mean')
        else:
            self.logger.warning(f"Unknown constrained_type: {self.constrained_type}. Defaulting to L1.")
            self.constrained = torch.nn.L1Loss(reduction='mean')
            self.criterion = torch.nn.L1Loss(reduction='mean')


    def save_checkpoint(self, epoch, batch_idx, is_best=False, routine=False):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_history': self.train_history,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            # Store relevant inference parameters if needed
            'inference_patch_size': self.inference_patch_size,
            'inference_stride': self.inference_stride,
            'gaussian_sigma': self.gaussian_sigma
        }

        # For routine saves (e.g., every epoch), save the current checkpoint
        checkpoint_dir_str = str(self.checkpoint_dir) # Convert Path to string
        if routine:
            checkpoint_path = os.path.join(checkpoint_dir_str, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            self.logger.debug(f'Routine checkpoint saved to {checkpoint_path}')
        else:
            # Remove previous latest_checkpoint if it exists
            for file in os.listdir(checkpoint_dir_str):
                if 'latest_checkpoint' in file:
                    try:
                        os.remove(os.path.join(checkpoint_dir_str, file))
                    except OSError as e:
                        self.logger.warning(f"Could not remove old latest checkpoint: {e}")
            checkpoint_path = os.path.join(checkpoint_dir_str, f'latest_checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
            torch.save(checkpoint, checkpoint_path)
            self.logger.debug(f'Latest checkpoint saved to {checkpoint_path}')

        # If it's the best model, save an extra copy
        if is_best:
            # Remove previous best_model if it exists
            for file in os.listdir(checkpoint_dir_str):
                if 'best_model' in file:
                     try:
                        os.remove(os.path.join(checkpoint_dir_str, file))
                     except OSError as e:
                        self.logger.warning(f"Could not remove old best model: {e}")
            best_model_path = os.path.join(checkpoint_dir_str, f'best_model_epoch_{epoch}_batch_{batch_idx}.pth')
            torch.save(checkpoint, best_model_path)
            self.logger.info(f'*** New best model saved to {best_model_path} (Loss: {self.best_loss:.6f}) ***')


    def load_checkpoint(self, checkpoint_path=None):
        """Load a checkpoint. Returns True if loaded, False otherwise."""
        checkpoint_dir_str = str(self.checkpoint_dir)
        load_path = None

        # Find the latest checkpoint
        if checkpoint_path: # Specific path provided
             if os.path.exists(checkpoint_path):
                 load_path = checkpoint_path
             else:
                 self.logger.warning(f"Provided checkpoint path does not exist: {checkpoint_path}")
                 return False
        else: 
            # Find automatically
            target_prefix = 'best_model' if self.test_mode else 'latest_checkpoint'
            found_files = [f for f in os.listdir(checkpoint_dir_str) if target_prefix in f]

            if not found_files:
                self.logger.info(f'No "{target_prefix}" checkpoint found in {checkpoint_dir_str}, starting from scratch or using initial model weights.')
                return False

            # Find the most recent one if multiple exist (e.g., multiple 'latest')
            found_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir_str, f)), reverse=True)

            # Find the latest checkpoint file(checkpoint_epoch_90, ..., checkpoint_epoch_100, choose the biggest one)
            found_files = [f for f in os.listdir(checkpoint_dir_str) if 'checkpoint_epoch' in f]
            if not found_files:
                self.logger.info(f'No checkpoint found in {checkpoint_dir_str}, starting from scratch or using initial model weights.')
                return False
            found_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True) # Sort by epoch number
            checkpoint_path = os.path.join(checkpoint_dir_str, found_files[0])

        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(Path(checkpoint_path))

            # Restore model and optimizer states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore training state
            self.start_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_loss = checkpoint['best_loss']
            self.train_history = checkpoint['train_history']

            # Restore random states
            random.setstate(checkpoint['random_state'])
            np.random.set_state(checkpoint['numpy_random_state'])
            torch.set_rng_state(checkpoint['torch_random_state'])
            if torch.cuda.is_available() and checkpoint['cuda_random_state'] is not None:
                torch.cuda.set_rng_state(checkpoint['cuda_random_state'])

            self.logger.info(f'Resumed training from epoch {self.start_epoch}')
            return True

        except FileNotFoundError:
            self.logger.error(f"Checkpoint file not found at {load_path}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading checkpoint from {load_path}: {e}")
            # Potentially fallback to starting fresh or raise error
            return False


# region Training
    def train(self):
        """Main training function."""
        if self.test_mode:
            print("--- Test Mode Active ---")
            if not self.load_checkpoint(): # Load best model for testing
                 self.logger.error("Failed to load checkpoint for testing. Aborting.")
                 return
            self._perform_testing(epoch=self.start_epoch -1) # Pass epoch for saving name
            return

        print("--- Training Mode Active ---")
        self.load_checkpoint()  # Attempt to load a previous checkpoint

        start_time = datetime.datetime.now()
        path = glob(os.path.join(self.train_data_path, '*.tif'))
        if not path:
             self.logger.error(f"No training files (*.tif) found in {self.train_data_path}")
             return
        batch_num = len(path) // self.train_batch_size

        self.logger.info(f"Starting training from epoch {self.start_epoch} to {self.epochs}")

        # Log training time for each epoch
        epoch_time = [0] * (self.epochs - self.start_epoch)

        try:
            for epoch in range(self.start_epoch, self.epochs):
                self.model.train()
                epoch_loss = 0.0

                # Log training time for each epoch
                epoch_start_time = datetime.datetime.now()

                for batch_idx, (inputs, labels) in enumerate(self.load_batch(self.train_data_path)):
                    inputs = torch.from_numpy(inputs).to(self.device, dtype=torch.float32)
                    labels = torch.from_numpy(labels).to(self.device, dtype=torch.float32)

                    # Forward pass and loss calculation
                    self.optimizer.zero_grad()
                    loss, loss_components = self._calculate_loss(inputs, labels)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    # Update training state
                    epoch_loss += loss.item()
                    self.global_step += 1

                    # Log training information
                    if batch_idx % 1 == 0:  # Log every batch
                        self._log_training_info(epoch, batch_idx, batch_num, loss.item(), loss_components, start_time)

                    # Save checkpoint periodically
                    if batch_idx % 100 == 0:  # Save every 100 batches
                        self.save_checkpoint(epoch, batch_idx)

                # End of epoch processing
                end_epoch_time = datetime.datetime.now()
                epoch_time[epoch - self.start_epoch] = (end_epoch_time - epoch_start_time).total_seconds()

                if batch_num > 0:
                    epoch_loss /= batch_num
                    self._end_epoch_processing(epoch, epoch_loss)
                else:
                    self.logger.warning(f"Epoch {epoch}: No batches were processed. Check data path and batch size.")

        except KeyboardInterrupt:
            self.logger.info('Training interrupted by user')
            self.save_checkpoint(epoch, batch_idx)

        except Exception as e:
            self.logger.exception(f'Training error occurred: {e}') # Log full traceback
            # Try saving checkpoint before raising
            try:
                self.save_checkpoint(epoch, batch_idx)
            except Exception as save_e:
                 self.logger.error(f"Failed to save checkpoint after error: {save_e}")
            raise # Re-raise the original exception

        finally:
            self.writer.close() # Ensure TensorBoard writer is closed
            self.logger.info("Training finished or stopped.")

            total_training_time = sum(epoch_time)
            self.logger.info(f"Total training time: {total_training_time/3600:.2f} hours")

            avg_epoch_time = sum(epoch_time) / len(epoch_time) if epoch_time else 0
            self.logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")

            # Log time for each epoch
            for i, t in enumerate(epoch_time):
                self.logger.info(f"Epoch {i + self.start_epoch}: {t:.2f} seconds")

# endregion


# region Loss Calculation
    def _calculate_loss(self, inputs, labels):
        """Calculate loss."""
        inputs_pred1 = self.model(inputs)
        loss_components = {}

        loss1 = self.criterion(inputs_pred1, labels)

        loss_components['loss1'] = loss1.item()

        if self.reg != 0:
            labels_pred1 = self.model(labels)
            loss2 = self.criterion(labels_pred1, inputs)
            loss3 = self.constrained(labels_pred1, inputs_pred1)

            loss_components['loss2'] = loss2.item()
            loss_components['loss3'] = loss3.item()

            if self.reg_sparse == 0:
                loss = (loss1 + loss2 + self.reg * loss3)/(2 + self.reg)
                loss_components['total_loss'] = loss.item()
            else:
                loss4 = self.criterion(torch.zeros_like(inputs_pred1), inputs_pred1) +\
                    self.criterion(labels_pred1, torch.zeros_like(labels_pred1))
                loss = (loss1 + loss2 + self.reg * loss3 + self.reg_sparse * loss4) \
                    / (2 + self.reg + 2 * self.reg_sparse)
                loss_components['total_loss'] = loss.item()
        else:
            if self.reg_sparse != 0:
                loss4 = self.criterion(torch.zeros_like(inputs_pred1), inputs_pred1)
                loss_components['loss4'] = loss4.item()
                loss = (loss1 + self.reg_sparse * loss4) / (1 + self.reg_sparse)
                loss_components['total_loss'] = loss.item()
            else:
                loss = loss1
                loss_components['total_loss'] = loss.item()

        # --- Visualization/Debugging ---
        if self.global_step % 100 == 0: # Visualize less frequently
            self._save_middle_image(inputs, labels, inputs_pred1,
                                   labels_pred1 if self.reg > 0 else None)

        return loss, loss_components
# endregion

    def _save_middle_image(self, inputs, labels, inputs_pred1, labels_pred1=None):
        """Saves a batch of intermediate images for debugging."""
        save_path = Path(f'./images/{self.tests_name}/middle_images')
        save_path.mkdir(parents=True, exist_ok=True)
        num_images_to_save = min(inputs.shape[0], 10) # Save first 4 images

        inputs_np = inputs.cpu().numpy()
        labels_np = labels.cpu().numpy()
        inputs_pred1_np = inputs_pred1.detach().cpu().numpy()
        if labels_pred1 is not None:
            labels_pred1_np = labels_pred1.detach().cpu().numpy()

        for i in range(num_images_to_save):
            fig, axes = plt.subplots(1, 4 if labels_pred1 is not None else 3, figsize=(15, 5))
            axes[0].imshow(inputs_np[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[0].set_title('Input')
            axes[0].axis('off')

            axes[1].imshow(labels_np[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Label')
            axes[1].axis('off')

            axes[2].imshow(inputs_pred1_np[i, 0], cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Input -> Pred')
            axes[2].axis('off')

            if labels_pred1 is not None:
                axes[3].imshow(labels_pred1_np[i, 0], cmap='gray', vmin=0, vmax=1)
                axes[3].set_title('Label -> Pred')
                axes[3].axis('off')

            plt.tight_layout()
            img_save_path = save_path / f'step_{self.global_step}_batch_{i}.png' # Save as PNG for easier viewing
            try:
                plt.savefig(img_save_path)
            except Exception as e:
                self.logger.warning(f"Could not save middle image: {e}")
            plt.close(fig) # Close the figure to free memory

    def _log_training_info(self, epoch, batch_idx, batch_num, loss, loss_components, start_time):
        """Log training information."""
        elapsed_time = datetime.datetime.now() - start_time
        eta = (elapsed_time / (batch_idx + 1)) * (batch_num - (batch_idx + 1)) if batch_idx > 0 else "N/A"
        log_msg = (
            f"Epoch {epoch}/{self.epochs} | Batch {batch_idx+1}/{batch_num} | "
            f"Loss: {loss:.5f} | LR: {self.optimizer.param_groups[0]['lr']:.1e} | "
            f"Elapsed: {str(elapsed_time).split('.')[0]} | ETA: {str(eta).split('.')[0]}"
        )
        self.logger.info(log_msg)

        # Log to TensorBoard
        self.writer.add_scalar('Loss/train_batch', loss, self.global_step)
        self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
        for name, value in loss_components.items():
             # Prefix component losses for clarity
             self.writer.add_scalar(f'Loss_Components/{name}', value, self.global_step)


# region End of Epoch Processing
    def _end_epoch_processing(self, epoch, epoch_loss):
        """Process operations at the end of each epoch."""
        self.logger.info(f"--- Epoch {epoch} finished. Average Loss: {epoch_loss:.6f} ---")
        self.writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        # Update learning rate
        if self.ifadaptive_lr:
            self.scheduler.step(epoch_loss)

        # Log current learning rate for the epoch
        self.writer.add_scalar('Learning_rate/epoch', self.optimizer.param_groups[0]['lr'], epoch)

        # Save the best model if loss improves
        is_best = epoch_loss < self.best_loss
        if is_best:
            self.best_loss = epoch_loss
            self.logger.info(f"*** New best loss achieved: {self.best_loss:.6f} ***")
            self.save_checkpoint(epoch, -1, is_best=True)

        # Routinely save model at the end of each epoch
        self.save_checkpoint(epoch, -1, routine=True)

        # Perform testing/validation
        if self.test_path and os.path.exists(self.test_path):
            self.logger.info(f"--- Running validation/testing for Epoch {epoch} ---")
            self._perform_testing(epoch)
        else:
            self.logger.warning(f"Test path '{self.test_path}' not found or not specified. Skipping validation.")

# endregion

# region Overlapping Patch Utilities
    def _generate_gaussian_weights(self, size, sigma):
        """Generates a 2D Gaussian kernel."""
        center = size / 2 - 0.5 # Center pixel coordinate
        y, x = np.ogrid[:size, :size]
        weights = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
        # Ensure weights don't become exactly zero (prevents potential division by zero)
        weights[weights < np.finfo(float).eps * weights.max()] = np.finfo(float).eps * weights.max()
        return torch.from_numpy(weights).float() # Return as torch tensor

    def _normalize_patch(self, patch_np):
        """Applies local min-max normalization to a patch (NumPy array)."""
        # Matches the 'local' normalization in load_batch
        min_val = np.min(patch_np)
        max_val = np.max(patch_np)
        range_val = max_val - min_val
        if range_val < 1e-8: # Avoid division by zero for flat patches
            # Return 0 or 0.5, depending on desired behavior for flat patches
            return np.zeros_like(patch_np, dtype=np.float32) if min_val == 0 else np.full_like(patch_np, 0.5, dtype=np.float32)
        normalized = (patch_np - min_val) / range_val
        return normalized.astype(np.float32)

    # Optional: De-normalization if needed, but weighted average handles scaling implicitly
    # def _denormalize_patch(self, patch_tensor, min_val, max_val):
    #     range_val = max_val - min_val
    #     if range_val < 1e-8:
    #          return torch.full_like(patch_tensor, min_val) # Return the constant value
    #     return patch_tensor * range_val + min_val

# endregion


# region Testing
    def _perform_testing(self, epoch):
        """Perform testing."""
        torch.cuda.empty_cache()
        patch_size = self.inference_patch_size
        stride = self.inference_stride
        weights = self._gaussian_weights.to(self.device)
        save_path_root = Path(f'./images/{self.tests_name}/images')
        save_path_root.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving test results for epoch {epoch} to {save_path_root}")

        with torch.no_grad():
            self.model.eval()
            for batch_idx, (data, batch_files) in enumerate(self.load_test_batch(self.test_path)):
                # Perform both original inference and overlapping patch inference
                full_img_np = data.copy()
                full_img_np = np.squeeze(full_img_np)
                img_path = batch_files[0]
                original_shape = full_img_np.shape

                self.logger.info(f"Processing test image: {os.path.basename(img_path)} (Shape: {original_shape})")
                
                # Overlapping patch inference
                # --- Basic Normalization ---
                # Normalize the full image to be in [0, 1] range
                full_img_np = self._normalize_patch(full_img_np)

                # --- Padding ---
                # Calculate padding needed for height and width
                pad_h = (stride - (original_shape[0] - patch_size) % stride) % stride
                pad_w = (stride - (original_shape[1] - patch_size) % stride) % stride

                # Pad the image using reflection padding
                # Add padding: (left, right, top, bottom)
                img_padded_np = np.pad(full_img_np, ((0, pad_h), (0, pad_w)), mode='reflect')
                img_padded = torch.from_numpy(img_padded_np).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32) # Add Batch and Channel dims
                padded_shape = img_padded.shape # (1, 1, H_pad, W_pad)

                # --- Accumulators ---
                result_accumulator = torch.zeros_like(img_padded, device=self.device)
                weight_accumulator = torch.zeros_like(img_padded, device=self.device)

                # --- Sliding Window ---
                for y in range(0, padded_shape[2] - patch_size + 1, stride):
                    for x in range(0, padded_shape[3] - patch_size + 1, stride):
                        # Extract patch (as tensor)
                        patch = img_padded[:, :, y:y+patch_size, x:x+patch_size] # (1, 1, P, P)

                        # --- Normalize Patch ---
                        # Convert patch to numpy for normalization function
                        patch_np = patch.squeeze().cpu().numpy()

                        if(self.inference_mode == 'local'):
                            # Local normalization: use patch min/max
                            normalized_patch_np = self._normalize_patch(patch_np)
                            # Convert back to tensor and add dims
                            normalized_patch = torch.from_numpy(normalized_patch_np).unsqueeze(0).unsqueeze(0).to(self.device)
                        else:
                            # Convert back to tensor and add dims
                            normalized_patch = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).to(self.device)

                        # --- Model Inference ---
                        prediction = self.model(normalized_patch) # (1, 1, P, P)

                        # --- Accumulate Results ---
                        result_accumulator[:, :, y:y+patch_size, x:x+patch_size] += prediction * weights
                        weight_accumulator[:, :, y:y+patch_size, x:x+patch_size] += weights

                # --- Final Image ---
                # Avoid division by zero: add small epsilon where weights are zero (shouldn't happen with Gaussian)
                final_image_padded = result_accumulator / (weight_accumulator + 1e-8)

                # --- Crop back to original size ---
                final_image = final_image_padded[:, :, :original_shape[0], :original_shape[1]]

                # --- Save Result ---
                final_image_np = final_image.squeeze().cpu().numpy() # (H, W)
                self.saveResult2(epoch, save_path_root, final_image_np, img_path) # Pass single image path

                # Original inference logic
                data = torch.from_numpy(data)
                data = data.to(self.device, dtype=torch.float32)
                test_pred = self.model(data)

                test_pred = test_pred.to(torch.device("cpu"))
                test_pred = test_pred.numpy()

                save_path = Path(f'./images/{self.tests_name}/images')
                save_path.mkdir(parents=True, exist_ok=True)
                self.saveResult(epoch, save_path, test_pred, batch_files)
        
        self.logger.info(f"--- Testing completed for epoch {epoch} ---")


    def saveResult(self, epoch, save_path, image_arr, batch_files):
        """Save result images."""
        for i, item in enumerate(image_arr):
            item = self.normalize(item)
            # Extract original filename and generate new name
            original_path = Path(batch_files[i])
            original_name = original_path.stem
            new_filename = f"{original_name}_epoch_{epoch}.tif"
            # Save image
            imsave(os.path.join(save_path, new_filename), item)
    
    
    def saveResult2(self, epoch, save_path, image_np, original_file_path):
        """Saves a single result image, applying display normalization."""
        # Apply 0-1 normalization for saving/display purposes
        # This might not be the actual intensity range if de-normalization was needed
        image_normalized_display = self.normalize(image_np)

        # Generate filename
        original_path = Path(original_file_path)
        original_name = original_path.stem
        # Suffix can be added by user if needed, e.g., '_denoised'

        if(self.inference_mode == 'local'):
            new_filename = f"{original_name}_epoch_{epoch}_local.tif"
        elif(self.inference_mode == 'global'):
            new_filename = f"{original_name}_epoch_{epoch}_global.tif"

        # new_filename = f"{original_name}_epoch_{epoch}.tif"
        save_full_path = save_path / new_filename

        try:
            # Ensure dtype is suitable for imsave (e.g., float32 or uint16 if scaled)
            # imsave might convert float32 to uint8 by default, check its behavior
            # Consider saving as float32 if precision is important:
            # imsave(save_full_path, image_normalized_display.astype(np.float32), check_contrast=False)
            imsave(save_full_path, image_normalized_display, check_contrast=False)
            self.logger.debug(f"Saved result: {save_full_path}")
        except Exception as e:
            self.logger.error(f"Failed to save image {save_full_path}: {e}")

# endregion

    def normalize(self, stack):
        """Normalizes image to [0, 1] float32 for display/saving."""
        stack = stack.astype(np.float32)
        min_val = np.min(stack)
        max_val = np.max(stack)
        range_val = max_val - min_val
        if range_val < 1e-8:
            # Handle flat images - return all zeros or min_val/max_val
            return np.zeros_like(stack) if min_val == 0 else np.full_like(stack, min_val)
        stack = (stack - min_val) / range_val
        return stack # Returns float32 in [0, 1]


    def load_batch(self, traindata_path, seed=123):
        """Load a batch of training data (Original logic, assumes patch-sized inputs)."""
        path = glob(os.path.join(traindata_path, '*.tif'))
        if not path:
             self.logger.error(f"No training files (*.tif) found in {traindata_path}")
             return # Stop iteration if no files

        batch_num = len(path) // self.train_batch_size
        # imsize = (self.train_batch_size, 1, self.img_res[0], self.img_res[1]) # Not used

        random.seed(seed)  # For reproducibility
        np.random.seed(seed)
        
        # Shuffle paths each epoch for better training
        random.shuffle(path)

        for i in range(batch_num):
            batch_indices = range(i * self.train_batch_size, (i + 1) * self.train_batch_size)
            batch_files = [path[k] for k in batch_indices]

            imgs_As = []
            imgs_Bs = []
            for img_file in batch_files:
                try:
                    img = imread(img_file)
                    h, w = img.shape

                    # Assuming concatenated input format (Input | Label)
                    if w != self.img_res[1] * 2 or h != self.img_res[0]:
                        self.logger.warning(f"Skipping {img_file}: Expected shape ({self.img_res[0]}, {self.img_res[1]*2}), got ({h}, {w})")
                        continue # Skip files with incorrect dimensions

                    half_w = self.img_res[1] # Use defined resolution

                    img_data = img[:, :half_w]
                    img_label = img[:, half_w:]

                    # --- Augmentation ---
                    # Horizontal Flip
                    if random.random() < 0.5:
                        img_data = np.fliplr(img_data)
                        img_label = np.fliplr(img_label)
                    # Vertical Flip
                    if random.random() < 0.5: # Changed from > 0.5 for consistency
                        img_data = np.flipud(img_data)
                        img_label = np.flipud(img_label)
                    # Rotation
                    rot_k = random.randint(0, 3) # 0, 1, 2, or 3 rotations by 90 degrees
                    if rot_k > 0:
                        img_data = np.rot90(img_data, rot_k)
                        img_label = np.rot90(img_label, rot_k)

                    # --- Normalization (Crucial Part) ---
                    # Ensure this matches the inference patch normalization
                    if self.work_mode == 'local':
                        # Apply local min-max normalization PER PATCH (img_data, img_label)
                        img_data = self._normalize_patch(img_data) # Use the same function
                        img_label = self._normalize_patch(img_label)
                    elif self.work_mode == 'global':
                        # Apply global scaling (e.g., divide by 255 or 65535)
                        # Ensure this scaling is appropriate for your data type
                        img_data = img_data / 255.0 # Example for 8-bit
                        img_label = img_label / 255.0 # Example for 8-bit
                        # img_data = img_data / 65535.0 # Example for 16-bit
                        # img_label = img_label / 65535.0 # Example for 16-bit
                    else:
                         self.logger.error(f"Unsupported work_mode: {self.work_mode}")
                         # Handle error appropriately

                    # Add channel dimension
                    img_data = img_data.reshape(1, self.img_res[0], self.img_res[1])
                    img_label = img_label.reshape(1, self.img_res[0], self.img_res[1])

                    imgs_As.append(img_data)
                    imgs_Bs.append(img_label)

                except Exception as e:
                    self.logger.error(f"Error loading or processing training file {img_file}: {e}")

            # Ensure we have a full batch before yielding (or handle partial last batch)
            if len(imgs_As) == self.train_batch_size:
                imgs_As_np = np.array(imgs_As, dtype=np.float32)
                imgs_Bs_np = np.array(imgs_Bs, dtype=np.float32)
                yield imgs_As_np, imgs_Bs_np
            elif i == batch_num -1 and len(imgs_As) > 0:
                 # Option to yield partial last batch if needed
                 # imgs_As_np = np.array(imgs_As, dtype=np.float32)
                 # imgs_Bs_np = np.array(imgs_Bs, dtype=np.float32)
                 # yield imgs_As_np, imgs_Bs_np
                 pass # Default: drop partial last batch

    # Modified test image loading logic to accommodate test images of different sizes
    def load_test_batch(self, test_path):
        path = glob(os.path.join(test_path, '*.tif'))
        if not path:
            self.logger.error(f"No test files (*.tif) found in {test_path}")
            raise ValueError("No .tif files found in test directory.")

        total_files = len(path)
        batch_size = self.test_batch_size
        batch_num = int(np.ceil(total_files / batch_size))

        for i in range(batch_num):
            start = i * batch_size
            end = min((i+1)*batch_size, total_files)
            current_batch_size = end - start
            batch = path[start:end]

            # Read the first image in the batch as a size reference
            img0 = imread(batch[0])
            h, w = img0.shape

            # Initialize batch array
            imgs_A = np.zeros((current_batch_size, 1, h, w), dtype=np.float32)

            # Process the first image
            img0 = (img0 - np.min(img0)) / (np.max(img0) - np.min(img0) + 1e-8)
            imgs_A[0] = img0.reshape(1, h, w)

            # Process remaining images
            for j in range(1, current_batch_size):
                img = imread(batch[j])
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
                imgs_A[j] = img.reshape(1, h, w)

            yield imgs_A, batch
