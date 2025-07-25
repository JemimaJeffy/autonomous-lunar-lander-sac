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
import shutil
from datetime import datetime
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
        
        # Get the current reward - FIXED: Check both possible reward sources
        reward_added = False
        if 'rewards' in self.locals and len(self.locals['rewards']) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
            reward_added = True
        elif hasattr(self.training_env, 'get_attr') and len(self.training_env.get_attr('reward_range')) > 0:
            # Alternative way to get rewards from vectorized env
            try:
                infos = self.locals.get('infos', [])
                if infos and len(infos) > 0 and 'reward' in infos[0]:
                    self.current_episode_reward += infos[0]['reward']
                    reward_added = True
            except:
                pass
        
        # Check if episode is done - IMPROVED: Multiple ways to detect episode end
        episode_done = False
        if 'dones' in self.locals and len(self.locals['dones']) > 0:
            episode_done = self.locals['dones'][0]
        elif 'infos' in self.locals and len(self.locals['infos']) > 0:
            # Check if episode ended via info dict
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_done = True
                # Get final episode reward from info if available
                if 'r' in info['episode']:
                    self.current_episode_reward = info['episode']['r']
                elif 'reward' in info['episode']:
                    self.current_episode_reward = info['episode']['reward']
        
        if episode_done:
            # Episode completed
            self._log_episode_completion()
            
            # FIXED: Always log to tensorboard, even if episode reward is 0
            self._force_log_episode()
            
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
    
    def _force_log_episode(self):
        """Force log episode data to ensure it gets recorded"""
        if hasattr(self.model, 'logger'):
            ep_reward = self.current_episode_reward
            ep_length = self.current_episode_length
            
            # Force record episode data
            self.model.logger.record("episode/reward", ep_reward)
            self.model.logger.record("episode/length", ep_length)
            self.model.logger.record("episode/count", self.episode_count)
            self.model.logger.record("episode/best_reward", self.best_reward)
            
            if len(self.recent_rewards) > 0:
                avg_reward_100 = np.mean(self.recent_rewards)
                self.model.logger.record("episode/avg_reward_100", avg_reward_100)
            
            if len(self.recent_losses) > 0:
                avg_loss = np.mean(self.recent_losses)
                self.model.logger.record("episode/avg_actor_loss", avg_loss)
            
            # Force dump the logs
            self.model.logger.dump(self.num_timesteps)
    
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
    
class AdaptiveHyperparameterCallback(BaseCallback):
    """Callback to gradually adjust SAC hyperparameters based on performance"""
    def __init__(self, verbose=0):
        super(AdaptiveHyperparameterCallback, self).__init__(verbose)
        self.SCORE_THRESHOLDS =  {1: (30, 80), 2: (80, 85), 3: (85, 90), 4: (90, 100)} 
        self.recent_rewards = []
        self.performance_history = []
        self.last_adjustment_step = 0
        self.adjustment_interval = 5000  # More frequent but smaller adjustments
        self.current_curriculum_level = 1
        self.stagnation_threshold = 2  # Reduced threshold
        self.episodes_since_improvement = 0
        self.best_recent_avg = float('-inf')
        self.improvement_threshold = 0.2  # Smaller threshold for detecting improvement
        self.initialization_period = 1  # NEW: Wait for this many rewards before first evaluation
        self.is_initialized = False  # NEW: Track if we've properly initialized
        
        # FIXED: Add episode tracking variables
        self.current_episode_reward = 0.0
        self.episode_rewards_buffer = []  # Buffer to collect episode rewards
        
        # Current hyperparameter state (starting values)
        self.current_params = {
            'learning_rate': 3e-4,
            'ent_coef': 0.1,  # Will be converted to float if 'auto'
            'tau': 0.005,
            'gradient_steps': 1
        }
        
        # Adjustment increments (much smaller changes)
        self.param_adjustments = {
            'learning_rate': {
                'increase': 1.2,    # 20% increase
                'decrease': 0.9,    # 10% decrease
                'min_val': 1e-5,
                'max_val': 1e-3
            },
            'ent_coef': {
                'increase': 1.3,    # 30% increase for exploration
                'decrease': 0.85,   # 15% decrease for exploitation
                'min_val': 0.001,
                'max_val': 0.5
            },
            'tau': {
                'increase': 1.5,    # Faster target updates
                'decrease': 0.8,    # Slower target updates
                'min_val': 0.001,
                'max_val': 0.02
            },
            'gradient_steps': {
                'increase': 1,      # +1 step
                'decrease': 0,      # -1 step (but min 1)
                'min_val': 1,
                'max_val': 4
            }
        }
        
        self.consecutive_stagnations = 0
        self.max_consecutive_adjustments = 3  # Limit consecutive adjustments
        
        # NEW: Add restart mechanism
        self.restart_requested = False
        self.new_hyperparams = None
        
    def _on_step(self) -> bool:
        # Check if restart was requested
        if self.restart_requested:
            return False  # Stop training to trigger restart
            
        # FIXED: Better episode reward tracking
        # Track current episode progress
        if 'rewards' in self.locals and len(self.locals['rewards']) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
        
        # FIXED: Better episode completion detection
        episode_done = False
        if 'dones' in self.locals and len(self.locals['dones']) > 0:
            episode_done = self.locals['dones'][0]
        elif 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_done = True
                # Get final episode reward from info if available
                if 'r' in info['episode']:
                    self.current_episode_reward = info['episode']['r']
                elif 'reward' in info['episode']:
                    self.current_episode_reward = info['episode']['reward']
        
        # FIXED: Store completed episode rewards
        if episode_done:
            self.episode_rewards_buffer.append(self.current_episode_reward)
            self.recent_rewards.append(self.current_episode_reward)
            
            # Keep only recent rewards for moving average
            if len(self.recent_rewards) > 100:
                self.recent_rewards.pop(0)
                
            # Reset for next episode
            self.current_episode_reward = 0.0
                
        # Check if it's time to evaluate and potentially adjust
        if (self.num_timesteps - self.last_adjustment_step) >= self.adjustment_interval and len(self.recent_rewards) >= 10:
            self._evaluate_and_adjust()
            self.last_adjustment_step = self.num_timesteps
            
        return True
    
    def _evaluate_and_adjust(self):
        """Evaluate current performance and make gradual adjustments if needed"""
        if len(self.recent_rewards) < 10:
            return
            
        # FIXED: Calculate recent average performance properly
        recent_avg = np.mean(self.recent_rewards[-50:]) if len(self.recent_rewards) >= 50 else np.mean(self.recent_rewards)
        
        # FIXED: Initialize baseline properly on first evaluation
        if not self.is_initialized:
            if len(self.recent_rewards) >= self.initialization_period:
                # Set initial baseline from accumulated data
                self.best_recent_avg = recent_avg - self.improvement_threshold  # Start slightly below current
                self.is_initialized = True
                print(f"[Step {self.num_timesteps}] Initialized baseline: {self.best_recent_avg:.2f}")
            else:
                # Still in initialization period, just log
                print(f"[Step {self.num_timesteps}] Initialization: {recent_avg:.2f} ({len(self.recent_rewards)}/{self.initialization_period})")
                return
        
        # Always log evaluation (MOVED after initialization check)
        print(f"[Step {self.num_timesteps}] Performance Check: {recent_avg:.2f} | Stagnation: {self.episodes_since_improvement}/{self.stagnation_threshold}")
        
        # Check for improvement (FIXED: Only after initialization)
        improvement_detected = recent_avg > self.best_recent_avg + self.improvement_threshold
        if improvement_detected:
            self.best_recent_avg = recent_avg
            self.episodes_since_improvement = 0
            self.consecutive_stagnations = 0
            print(f"✅ Improvement detected! New best: {self.best_recent_avg:.2f}")
        else:
            self.episodes_since_improvement += 1
            
        # Store performance history
        self.performance_history.append({
            'step': self.num_timesteps,
            'avg_reward': recent_avg,
            'episodes_since_improvement': self.episodes_since_improvement
        })
        
        # Determine if adjustment is needed
        if self.episodes_since_improvement >= self.stagnation_threshold and self.consecutive_stagnations < self.max_consecutive_adjustments:
            self._prepare_restart_with_new_hyperparams(recent_avg)
            self.consecutive_stagnations += 1
        elif improvement_detected and self.consecutive_stagnations > 0:
            # Reset consecutive adjustments counter on improvement
            self.consecutive_stagnations = 0
            
    def _prepare_restart_with_new_hyperparams(self, recent_avg):
        """Prepare new hyperparameters and request restart"""
        adjustments_made = []
        new_params = self.current_params.copy()
        
        # Determine adjustment strategy based on performance level
        if recent_avg < -3:  # Very poor performance
            # Increase exploration gradually
            adjustments_made.extend(self._adjust_parameter_copy('ent_coef', 'increase', new_params))
            adjustments_made.extend(self._adjust_parameter_copy('learning_rate', 'increase', new_params))
            strategy = "POOR PERFORMANCE - Increasing exploration"
            
        elif recent_avg < -1:  # Below average performance  
            # Moderate exploration increase
            adjustments_made.extend(self._adjust_parameter_copy('ent_coef', 'increase', new_params))
            adjustments_made.extend(self._adjust_parameter_copy('tau', 'increase', new_params))
            strategy = "BELOW AVERAGE - Moderate exploration boost"
            
        elif recent_avg < 2:  # Average performance, likely stagnant
            # Balanced adjustment - try different combinations
            if self.consecutive_stagnations % 2 == 0:
                adjustments_made.extend(self._adjust_parameter_copy('ent_coef', 'increase', new_params))
                adjustments_made.extend(self._adjust_parameter_copy('gradient_steps', 'increase', new_params))
            else:
                adjustments_made.extend(self._adjust_parameter_copy('learning_rate', 'increase', new_params))
                adjustments_made.extend(self._adjust_parameter_copy('tau', 'decrease', new_params))
            strategy = "STAGNANT - Mixed adjustments"
            
        else:  # Good performance, fine-tune
            # Small adjustments for fine-tuning
            adjustments_made.extend(self._adjust_parameter_copy('learning_rate', 'decrease', new_params))
            adjustments_made.extend(self._adjust_parameter_copy('ent_coef', 'decrease', new_params))
            strategy = "GOOD PERFORMANCE - Fine-tuning"
        
        if adjustments_made:
            self._log_adjustment(recent_avg, strategy, adjustments_made)
            self.current_params = new_params
            self.new_hyperparams = new_params.copy()
            self.restart_requested = True
            
            # Save state for restart
            self._save_restart_state()
            
    def _adjust_parameter_copy(self, param_name, direction, params_dict):
        """Adjust a parameter in a copy of the parameters dictionary"""
        if param_name not in params_dict:
            return []
            
        old_value = params_dict[param_name]
        adjustment_config = self.param_adjustments[param_name]
        
        if direction == 'increase':
            if param_name == 'gradient_steps':
                new_value = min(old_value + adjustment_config['increase'], adjustment_config['max_val'])
            else:
                new_value = min(old_value * adjustment_config['increase'], adjustment_config['max_val'])
        else:  # decrease
            if param_name == 'gradient_steps':
                new_value = max(old_value - 1, adjustment_config['min_val'])
            else:
                new_value = max(old_value * adjustment_config['decrease'], adjustment_config['min_val'])
        
        # Only update if there's a meaningful change
        if abs(new_value - old_value) > (old_value * 0.05 if param_name != 'gradient_steps' else 0):
            params_dict[param_name] = new_value
            return [f"{param_name}: {old_value:.6f} → {new_value:.6f}"]
        
        return []
    
    def _save_restart_state(self):
        """Save the current state for restart"""
        restart_state = {
            'recent_rewards': self.recent_rewards,
            'performance_history': self.performance_history,
            'current_params': self.current_params,
            'best_recent_avg': self.best_recent_avg,
            'episodes_since_improvement': self.episodes_since_improvement,
            'consecutive_stagnations': self.consecutive_stagnations,
            'is_initialized': self.is_initialized,
            'current_timestep': self.num_timesteps
        }
        
        import pickle
        with open('restart_state.pkl', 'wb') as f:
            pickle.dump(restart_state, f)
            
    def load_restart_state(self):
        """Load state from a previous restart"""
        try:
            import pickle
            with open('restart_state.pkl', 'rb') as f:
                restart_state = pickle.load(f)
                
            self.recent_rewards = restart_state['recent_rewards']
            self.performance_history = restart_state['performance_history']
            self.current_params = restart_state['current_params']
            self.best_recent_avg = restart_state['best_recent_avg']
            self.episodes_since_improvement = restart_state['episodes_since_improvement']
            self.consecutive_stagnations = restart_state['consecutive_stagnations']
            self.is_initialized = restart_state['is_initialized']
            
            print(f"✅ Loaded restart state from timestep {restart_state['current_timestep']}")
            return True
        except FileNotFoundError:
            print("No restart state found, starting fresh")
            return False
        except Exception as e:
            print(f"Error loading restart state: {e}")
            return False
            
    def _log_adjustment(self, recent_avg, strategy, adjustments_made):
        """Log the hyperparameter adjustment"""
        print(f"\n{'='*50}")
        print(f"HYPERPARAMETER RESTART REQUESTED")
        print(f"{'='*50}")
        print(f"Performance: {recent_avg:.2f}")
        print(f"Strategy: {strategy}")
        print(f"Stagnation Count: {self.episodes_since_improvement}")
        print(f"Consecutive Adjustments: {self.consecutive_stagnations}")
        print(f"Timestep: {self.num_timesteps}")
        print(f"Adjustments made:")
        for adjustment in adjustments_made:
            print(f"  • {adjustment}")
        print(f"{'='*50}\n")
        
        # Log to tensorboard
        if hasattr(self.model, 'logger'):
            self.model.logger.record("adaptive/recent_avg_reward", recent_avg)
            self.model.logger.record("adaptive/episodes_since_improvement", self.episodes_since_improvement)
            self.model.logger.record("adaptive/consecutive_stagnations", self.consecutive_stagnations)
            for param, value in self.current_params.items():
                self.model.logger.record(f"adaptive/{param}", value)

    def reset_restart_flag(self):
        """Reset the restart flag after processing"""
        self.restart_requested = False
        self.new_hyperparams = None

    def _save_restart_state(self):
        """Save the current state for restart"""
        
        # Try to get current curriculum info
        current_curriculum_level = 1
        current_shaping_scale = 1.0
        
        try:
            if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                base_env = self.training_env.envs[0]
                if hasattr(base_env, 'env'):
                    base_env = base_env.env
                
                if hasattr(base_env, 'curriculum_level'):
                    current_curriculum_level = base_env.curriculum_level
                    current_shaping_scale = base_env.shaping_scale
        except:
            pass
        
        restart_state = {
            'recent_rewards': self.recent_rewards,
            'performance_history': self.performance_history,
            'current_params': self.current_params,
            'best_recent_avg': self.best_recent_avg,
            'episodes_since_improvement': self.episodes_since_improvement,
            'consecutive_stagnations': self.consecutive_stagnations,
            'is_initialized': self.is_initialized,
            'current_timestep': self.num_timesteps,
            'curriculum_level': current_curriculum_level,  # Add curriculum state
            'shaping_scale': current_shaping_scale         # Add shaping scale
        }
        
        import pickle
        with open('restart_state.pkl', 'wb') as f:
            pickle.dump(restart_state, f)

def create_unique_log_dir(base_log_dir):
    """Create a unique log directory for each restart"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_log_dir = f"{base_log_dir}_{timestamp}"
    os.makedirs(unique_log_dir, exist_ok=True)
    return unique_log_dir

def get_tensorboard_log_dir(base_log_dir, restart_count=0):
    """Get appropriate tensorboard log directory"""
    if restart_count == 0:
        return base_log_dir
    else:
        return f"{base_log_dir}/restart_{restart_count}"

def restart_training_with_new_hyperparams(model_dir, base_log_dir, total_timesteps, env_kwargs, adaptive_callback, restart_count=0):
    """Restart training with new hyperparameters"""
    
    # Kill existing Unreal processes
    print("Restarting training with new hyperparameters...")
    print("Cleaning up existing Unreal processes...")
    kill_existing_unreal_processes()
    time.sleep(5)
    
    # Find the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(model_dir)
    if not latest_checkpoint:
        print("❌ No checkpoint found for restart!")
        return None
        
    print(f"Loading checkpoint for restart: {latest_checkpoint}")
    
    # IMPORTANT: Preserve curriculum state from existing environment before closing
    current_curriculum_level = 1  # Default fallback
    current_shaping_scale = 1.0   # Default fallback
    
    try:
        # Try to get current curriculum state from the callback's recent performance
        if hasattr(adaptive_callback, 'recent_rewards') and len(adaptive_callback.recent_rewards) > 0:
            recent_avg = np.mean(adaptive_callback.recent_rewards[-50:])
            
            # Determine appropriate curriculum level based on performance
            if recent_avg >= 100:
                current_curriculum_level = 4
                current_shaping_scale = 0.1
            elif recent_avg >= 90:
                current_curriculum_level = 3
                current_shaping_scale = 0.3
            elif recent_avg >= 80:
                current_curriculum_level = 2
                current_shaping_scale = 0.5
            elif recent_avg >= 30:
                current_curriculum_level = 1
                current_shaping_scale = 1.0
            else:
                current_curriculum_level = 1  # Stay at easiest level for poor performance
                current_shaping_scale = 1.0
                
            print(f"🎯 Setting curriculum level {current_curriculum_level} based on performance {recent_avg:.2f}")
            
    except Exception as e:
        print(f"Warning: Could not determine curriculum level: {e}")
        current_curriculum_level = 1
        current_shaping_scale = 1.0
    
    # Create new environment with preserved curriculum settings
    env_kwargs_restart = env_kwargs.copy()
    env_kwargs_restart['max_reset_attempts'] = 5
    # Add curriculum parameters to environment kwargs
    env_kwargs_restart['initial_curriculum_level'] = current_curriculum_level
    env_kwargs_restart['initial_shaping_scale'] = current_shaping_scale
    
    def make_env():
        base_env = UnrealLunarLanderEnv(**env_kwargs_restart)
        # Explicitly set curriculum after creation if the env doesn't support init params
        if hasattr(base_env, 'curriculum_level'):
            base_env.curriculum_level = current_curriculum_level
            base_env.shaping_scale = current_shaping_scale
            print(f"✅ Set curriculum level: {current_curriculum_level}, shaping scale: {current_shaping_scale}")
        return ActionAdapter(base_env)
    
    env = DummyVecEnv([make_env])
    
    # IMPORTANT: Reset the environment and wait for proper connection
    print("Resetting environment and waiting for Unreal connection...")
    try:
        env.reset()
        time.sleep(3)
        
        # Verify curriculum settings after reset
        if hasattr(env.envs[0], 'env') and hasattr(env.envs[0].env, 'curriculum_level'):
            actual_level = env.envs[0].env.curriculum_level
            actual_scale = env.envs[0].env.shaping_scale
            print(f"📊 Verified curriculum - Level: {actual_level}, Scale: {actual_scale}")
        
        # Test environment response
        test_action = env.action_space.sample()
        env.step([test_action])
        print("✅ Environment test successful")
        
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        env.close()
        return None
    
    # ... rest of the function remains the same
    
    # Create unique log directory for this restart
    log_dir = get_tensorboard_log_dir(base_log_dir, restart_count)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create new SAC model with updated hyperparameters
    new_sac_kwargs = {
        'learning_rate': adaptive_callback.current_params['learning_rate'],
        'buffer_size': 1000000,
        'learning_starts': 100,  # Reduced further for restart
        'batch_size': 256,
        'tau': adaptive_callback.current_params['tau'],
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': int(adaptive_callback.current_params['gradient_steps']),
        'ent_coef': adaptive_callback.current_params['ent_coef'],
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
        'tensorboard_log': log_dir,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    try:
        # Load the checkpoint
        model = SAC.load(latest_checkpoint, env=env)
        
        # Update hyperparameters
        model.learning_rate = new_sac_kwargs['learning_rate']
        model.tau = new_sac_kwargs['tau']
        model.gradient_steps = new_sac_kwargs['gradient_steps']
        model.ent_coef = new_sac_kwargs['ent_coef']
        
        # IMPORTANT: Set up NEW logger with unique directory
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)
        
        # Log the restart event
        model.logger.record("restart/restart_count", restart_count)
        model.logger.record("restart/new_learning_rate", new_sac_kwargs['learning_rate'])
        model.logger.record("restart/new_tau", new_sac_kwargs['tau'])
        model.logger.record("restart/new_gradient_steps", new_sac_kwargs['gradient_steps'])
        model.logger.record("restart/new_ent_coef", new_sac_kwargs['ent_coef'])
        model.logger.dump(model.num_timesteps)
        
        print("✅ Model loaded and hyperparameters updated successfully!")
        print(f"📊 New logs will be saved to: {log_dir}")
        
        # Create fresh callback with loaded state
        new_adaptive_callback = AdaptiveHyperparameterCallback(verbose=1)
        new_adaptive_callback.load_restart_state()
        new_adaptive_callback.restart_requested = False
        
        # CRITICAL: Reset the environment one more time before returning
        env.reset()
        time.sleep(2)
        
        return model, env, new_adaptive_callback
        
    except Exception as e:
        print(f"❌ Error during restart: {e}")
        env.close()
        return None

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
#existing main
def main():
    # Configuration
    TOTAL_TIMESTEPS = 2000000
    CHECKPOINT_FREQ = 5000
    BASE_LOG_DIR = "logs/sac_lunar_lander_2"  # Base directory for logs
    MODEL_DIR = "models/sac_lunar_lander"
    
    # Create directories
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
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
    
    # Environment kwargs
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
    sac_kwargs = {
        'learning_rate': 3e-4,
        'buffer_size': 1000000,
        'learning_starts': 25000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'ent_coef': 'auto',
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
        'tensorboard_log': BASE_LOG_DIR,  # Use BASE_LOG_DIR initially
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
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
    
    # Create adaptive callback (initialize once)
    adaptive_callback = AdaptiveHyperparameterCallback(verbose=1)
    
    # Main training loop with restart capability
    remaining_timesteps = TOTAL_TIMESTEPS
    restart_count = 0
    
    if checkpoint_path:
        print(f"Continuing training from checkpoint for {TOTAL_TIMESTEPS} additional timesteps...")
    else:
        print(f"Starting fresh training for {TOTAL_TIMESTEPS} timesteps...")
    
    print(f"Checkpoints will be saved every {CHECKPOINT_FREQ} steps to {MODEL_DIR}")
    print(f"Logs will be saved to {BASE_LOG_DIR}")
    print("Press Ctrl+C to stop training and save model gracefully")
    print("\n" + "="*80)
    
    # Replace the entire while loop section with this:
    while remaining_timesteps > 0:
        try:
            # Get current log directory for this training session
            current_log_dir = get_tensorboard_log_dir(BASE_LOG_DIR, restart_count)
            
            # Set up logging for current session
            new_logger = configure(current_log_dir, ["stdout", "csv", "tensorboard"])
            model.set_logger(new_logger)
            
            # Reset the restart flag before training
            adaptive_callback.restart_requested = False
            
            # IMPORTANT: For restarts, do a fresh environment reset
            if restart_count > 0:
                print("Performing fresh environment reset after restart...")
                env.reset()
                time.sleep(2)
            
            # Create callbacks for this session
            callbacks = [
                DetailedLoggingCallback(verbose=1),
                adaptive_callback,
                CheckpointCallback(
                    save_freq=CHECKPOINT_FREQ,
                    save_path=MODEL_DIR,
                    name_prefix='sac_lunar_lander_2'
                ),
                GracefulSaveCallback(save_path=MODEL_DIR, verbose=1)
            ]
            
            print(f"📊 Training logs for this session: {current_log_dir}")
            print(f"🔄 Restart count: {restart_count}")
            
            # Start training
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False if checkpoint_path else True
            )
            
            # Check if training stopped due to restart request
            if hasattr(adaptive_callback, 'restart_requested') and adaptive_callback.restart_requested:
                print("🔄 Hyperparameter adjustment requested - restarting training...")
                
                # Calculate remaining timesteps
                remaining_timesteps = TOTAL_TIMESTEPS - model.num_timesteps
                restart_count += 1
                
                if remaining_timesteps <= 0:
                    print("✅ Training completed during restart request!")
                    break
                
                # Close current environment
                env.close()
                time.sleep(3)  # Wait for proper cleanup
                
                # Restart with new hyperparameters
                restart_result = restart_training_with_new_hyperparams(
                    MODEL_DIR, 
                    BASE_LOG_DIR, 
                    remaining_timesteps, 
                    env_kwargs, 
                    adaptive_callback, 
                    restart_count
                )
                
                if restart_result is None:
                    print("❌ Failed to restart training")
                    break
                    
                model, env, adaptive_callback = restart_result
                print(f"🔄 Training restarted with {remaining_timesteps} timesteps remaining")
                
                # Continue the while loop to start training again
                continue
            else:
                # If we reach here, training completed normally
                break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            break
        except Exception as e:
            print(f"Training stopped due to error: {e}")
            final_save_path = f"{MODEL_DIR}/error_recovery_model"
            model.save(final_save_path)
            print(f"Model saved to {final_save_path}")
            break
    
    # Save final model if training completed successfully
    if remaining_timesteps <= 0:
        final_save_path = f"{MODEL_DIR}/final_model"
        model.save(final_save_path)
        print(f"Training completed! Final model saved to {final_save_path}")
    
    # Final cleanup
    print("Cleaning up...")
    if 'env' in locals():
        env.close()
        print("Environment closed")
    
    # Create a summary of all training sessions
    create_training_summary(BASE_LOG_DIR, restart_count)
    print("Training completed!")

def create_training_summary(base_log_dir, restart_count):
    """Create a summary file of all training sessions"""
    summary_file = os.path.join(base_log_dir, "training_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write(f"Training Summary\n")
        f.write(f"================\n\n")
        f.write(f"Total restart count: {restart_count}\n")
        f.write(f"Base log directory: {base_log_dir}\n\n")
        
        f.write("Log directories:\n")
        f.write(f"  - Initial training: {base_log_dir}\n")
        
        for i in range(1, restart_count + 1):
            restart_dir = f"{base_log_dir}/restart_{i}"
            f.write(f"  - Restart {i}: {restart_dir}\n")
        
        f.write(f"\nTo view in TensorBoard:\n")
        f.write(f"  tensorboard --logdir {base_log_dir}\n")
        f.write(f"  This will show all training sessions with different colored lines\n")
    
    print(f"📋 Training summary saved to: {summary_file}")

if __name__ == "__main__":
    main()