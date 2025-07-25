import os
import sys
import time
import signal
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import torch
import subprocess
import platform
from pathlib import Path
import psutil

# Import your custom environment
from lunar_lander_env import UnrealLunarLanderEnv

class ActionAdapter(gym.ActionWrapper):
    """Adapter to convert 3D action to 6D action by adding zeros"""
    def __init__(self, env):
        super(ActionAdapter, self).__init__(env)
        # Change action space to 3D
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
    
    def action(self, action):
        # Convert 3D action to 6D by appending zeros
        extended_action = np.zeros(6, dtype=np.float32)
        extended_action[:3] = action
        return extended_action

class DetailedLoggingCallback(BaseCallback):
    """Custom callback for detailed episode logging"""
    def __init__(self, verbose=0):
        super(DetailedLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.recent_rewards = []
        self.recent_losses = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Track current episode progress
        self.current_episode_length += 1
        
        # Get the current reward
        if 'rewards' in self.locals:
            rewards = self.locals['rewards']
            if len(rewards) > 0:
                self.current_episode_reward += rewards[0]
        
        # Check if episode is done
        if 'dones' in self.locals:
            dones = self.locals['dones']
            if len(dones) > 0 and dones[0]:
                # Episode completed
                self._log_episode_completion()
                
                # Reset for next episode
                self.current_episode_reward = 0.0
                self.current_episode_length = 0
        
        # Log training metrics if available
        if hasattr(self.model, 'logger') and len(self.model.logger.name_to_value) > 0:
            # Get recent loss values
            if 'train/actor_loss' in self.model.logger.name_to_value:
                actor_loss = self.model.logger.name_to_value['train/actor_loss']
                self.recent_losses.append(actor_loss)
                if len(self.recent_losses) > 100:
                    self.recent_losses.pop(0)
        
        return True
    
    def _log_episode_completion(self):
        """Log detailed episode completion information"""
        ep_reward = self.current_episode_reward
        ep_length = self.current_episode_length
        
        self.episode_rewards.append(ep_reward)
        self.episode_lengths.append(ep_length)
        self.recent_rewards.append(ep_reward)
        self.episode_count += 1
        
        # Keep only recent rewards for moving average
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        
        # Update best reward
        if ep_reward > self.best_reward:
            self.best_reward = ep_reward
        
        # Calculate statistics
        avg_reward_100 = np.mean(self.recent_rewards) if self.recent_rewards else ep_reward
        avg_loss = np.mean(self.recent_losses) if self.recent_losses else 0.0
        
        # Print detailed episode information
        print(f"\n{'='*80}")
        print(f"EPISODE {self.episode_count} COMPLETED")
        print(f"{'='*80}")
        print(f"Episode Reward: {ep_reward:.2f}")
        print(f"Episode Length: {ep_length}")
        print(f"Best Reward Ever: {self.best_reward:.2f}")
        print(f"Avg Reward (last 100): {avg_reward_100:.2f}")
        print(f"Avg Actor Loss (recent): {avg_loss:.6f}")
        print(f"Total Episodes: {self.episode_count}")
        print(f"Total Steps: {self.num_timesteps}")
        
        # Episode status based on reward
        if ep_reward > 50:
            status = "🟢 EXCELLENT"
        elif ep_reward > 0:
            status = "🟡 GOOD"
        elif ep_reward > -50:
            status = "🟠 POOR"
        else:
            status = "🔴 TERRIBLE"
        print(f"Performance: {status}")
        
        # Curriculum information if available
        try:
            if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                base_env = self.training_env.envs[0]
                # Check if it's wrapped (ActionAdapter)
                if hasattr(base_env, 'env'):
                    base_env = base_env.env
                
                if hasattr(base_env, 'curriculum_level'):
                    curriculum_level = base_env.curriculum_level
                    shaping_scale = base_env.shaping_scale
                    print(f"Curriculum Level: {curriculum_level}")
                    print(f"Shaping Scale: {shaping_scale:.3f}")
                    
                    # Update curriculum with average reward
                    if hasattr(base_env, 'set_curriculum_level'):
                        base_env.set_curriculum_level(avg_reward_100)
        except Exception as e:
            print(f"Could not access curriculum info: {e}")
        
        print(f"{'='*80}\n")
        
        # Log to tensorboard
        if hasattr(self, 'logger'):
            self.logger.record("episode/reward", ep_reward)
            self.logger.record("episode/length", ep_length)
            self.logger.record("episode/best_reward", self.best_reward)
            self.logger.record("episode/avg_reward_100", avg_reward_100)
            if avg_loss > 0:
                self.logger.record("episode/avg_actor_loss", avg_loss)

class GracefulSaveCallback(BaseCallback):
    """Callback to handle graceful shutdown and saving"""
    def __init__(self, save_path, verbose=0):
        super(GracefulSaveCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_requested = False
        
        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        if platform.system() != "Windows":
            signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\n\nReceived signal {signum}. Requesting graceful shutdown...")
        self.save_requested = True
    
    def _on_step(self) -> bool:
        if self.save_requested:
            print("Saving model before shutdown...")
            save_path = f"{self.save_path}/final_model_step_{self.num_timesteps}"
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
            return False  # Stop training
        return True

def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint in the model directory"""
    if not os.path.exists(model_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(model_dir):
        if file.endswith('.zip') and 'sac_lunar_lander' in file:
            checkpoint_files.append(file)
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time to get the latest
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    return os.path.join(model_dir, checkpoint_files[0])

def find_unreal_executable():
    """Try to find Unreal Engine executable"""
    possible_paths = [
        "C:/Users/jemim/VSCode Projects/Lunar Lander 10/Unreal Engine/lunar_lander.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def kill_existing_unreal_processes():
    """Kill any existing Unreal Engine processes"""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if 'unreal' in proc.info['name'].lower() or 'ue' in proc.info['name'].lower():
                print(f"Killing existing Unreal process: {proc.info['name']} (PID: {proc.info['pid']})")
                try:
                    proc.kill()
                except:
                    pass
    except Exception as e:
        print(f"Error killing existing processes: {e}")

def get_hyperparameter_mode():
    """Get hyperparameter mode from user input"""
    print("\n" + "="*60)
    print("SELECT HYPERPARAMETER MODE:")
    print("="*60)
    print("1. EXPLORATION BOOST - High exploration, lower learning rate")
    print("2. BALANCED - Standard balanced parameters")
    print("3. FINETUNING - Lower exploration, higher learning rate")
    print("="*60)
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            else:
                print("Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

def get_sac_hyperparameters(mode):
    """Get SAC hyperparameters based on mode"""
    base_kwargs = {
        'buffer_size': 1000000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'target_update_interval': 1,
        'target_entropy': 'auto',
        'use_sde': False,
        'sde_sample_freq': -1,
        'use_sde_at_warmup': False,
        'policy_kwargs': dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU,
        ),
        'verbose': 1,
        'tensorboard_log': 'sac_lunar_lander_manual',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if mode == 1:  # EXPLORATION BOOST
        mode_kwargs = {
            'learning_rate': 1e-4,      # Lower learning rate for stability
            'learning_starts': 15000,   # Start learning earlier
            'ent_coef': 0.2,           # Higher entropy for more exploration
        }
        mode_name = "EXPLORATION BOOST"
    elif mode == 2:  # BALANCED
        mode_kwargs = {
            'learning_rate': 3e-4,      # Standard learning rate
            'learning_starts': 25000,   # Standard start
            'ent_coef': 'auto',        # Auto entropy coefficient
        }
        mode_name = "BALANCED"
    elif mode == 3:  # FINETUNING
        mode_kwargs = {
            'learning_rate': 5e-4,      # Higher learning rate for faster convergence
            'learning_starts': 35000,   # More initial exploration
            'ent_coef': 0.01,          # Lower entropy for exploitation
        }
        mode_name = "FINETUNING"
    
    # Merge base and mode-specific kwargs
    final_kwargs = {**base_kwargs, **mode_kwargs}
    
    print(f"\n🎯 Using {mode_name} mode:")
    print(f"   Learning Rate: {final_kwargs['learning_rate']}")
    print(f"   Learning Starts: {final_kwargs['learning_starts']}")
    print(f"   Entropy Coefficient: {final_kwargs['ent_coef']}")
    print("")
    
    return final_kwargs

def main():
    # Configuration
    TOTAL_TIMESTEPS = 2000000  # Adjust as needed
    CHECKPOINT_FREQ = 5000
    LOG_DIR = "logs/sac_lunar_lander_manual"
    MODEL_DIR = "models/sac_lunar_lander"
    
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check for command line arguments for checkpoint path
    checkpoint_path = None
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file {checkpoint_path} does not exist!")
            return
    else:
        # Try to find the latest checkpoint automatically
        checkpoint_path = find_latest_checkpoint(MODEL_DIR)
        if checkpoint_path:
            print(f"Found latest checkpoint: {checkpoint_path}")
            user_input = input("Do you want to continue from this checkpoint? (y/n): ").strip().lower()
            if user_input != 'y':
                checkpoint_path = None
    
    # Kill existing Unreal processes
    print("Cleaning up existing Unreal processes...")
    kill_existing_unreal_processes()
    time.sleep(2)
    
    # Find Unreal executable
    unreal_exe = find_unreal_executable()
    if unreal_exe:
        print(f"Found Unreal Engine at: {unreal_exe}")
    else:
        print("Warning: Unreal Engine executable not found. Please specify manually.")
        unreal_exe = input("Enter path to Unreal Engine executable (or press Enter to continue without auto-launch): ").strip()
        if not unreal_exe:
            unreal_exe = None
    
    # Create environment
    print("Creating environment...")
    env_kwargs = {
        'flask_port': 5000,
        'host': '0.0.0.0',
        'unreal_exe_path': unreal_exe,
        'launch_unreal': unreal_exe is not None,
        'ue_launch_args': ['-game', '-ResX=800', '-ResY=600', '-windowed'],
        'max_reset_attempts': 3
    }
    
    # Wrap environment with action adapter
    def make_env():
        base_env = UnrealLunarLanderEnv(**env_kwargs)
        return ActionAdapter(base_env)
    
    env = DummyVecEnv([make_env])
    
    # SAC Hyperparameters (optimized for continuous control)
    hp_mode = get_hyperparameter_mode()
    
    # Get SAC hyperparameters based on selected mode
    sac_kwargs = get_sac_hyperparameters(hp_mode)
    
    print(f"Using device: {sac_kwargs['device']}")
    
    # Create or load SAC model
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        try:
            # First, try loading without specifying kwargs to avoid conflicts
            model = SAC.load(checkpoint_path, env=env)
            print("✅ Model loaded successfully!")
            
            # Update the environment and logger
            model.set_env(env)
            
            # Get the number of timesteps from the checkpoint filename if possible
            import re
            match = re.search(r'(\d+)_steps', checkpoint_path)
            if match:
                loaded_steps = int(match.group(1))
                print(f"Loaded model trained for {loaded_steps} steps")
            else:
                print("Could not determine number of steps from filename")
                
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Creating new model instead...")
            model = SAC('MlpPolicy', env, **sac_kwargs)
    else:
        print("Creating new SAC model...")
        model = SAC('MlpPolicy', env, **sac_kwargs)
    
    # Set up logging
    new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Create callbacks
    callbacks = [
        DetailedLoggingCallback(verbose=1),
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=MODEL_DIR,
            name_prefix='sac_lunar_lander_manual'
        ),
        GracefulSaveCallback(save_path=MODEL_DIR, verbose=1)
    ]
    
    if checkpoint_path:
        print(f"Continuing training from checkpoint for {TOTAL_TIMESTEPS} additional timesteps...")
    else:
        print(f"Starting fresh training for {TOTAL_TIMESTEPS} timesteps...")
    
    print(f"Checkpoints will be saved every {CHECKPOINT_FREQ} steps to {MODEL_DIR}")
    print(f"Logs will be saved to {LOG_DIR}")
    print("Press Ctrl+C to stop training and save model gracefully")
    print("\n" + "="*80)
    
    try:
        # Start training
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False if checkpoint_path else True  # Don't reset if continuing
        )
        
        # Save final model
        final_save_path = f"{MODEL_DIR}/final_model"
        model.save(final_save_path)
        print(f"Training completed! Final model saved to {final_save_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training stopped due to error: {e}")
        final_save_path = f"{MODEL_DIR}/error_recovery_model"
        model.save(final_save_path)
        print(f"Model saved to {final_save_path}")
    finally:
        # Clean up
        print("Cleaning up...")
        env.close()
        print("Environment closed")

if __name__ == "__main__":
    main()