import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    def __init__(self, log_dir="logs/sac_lunar_lander"):
        self.log_dir = Path(log_dir)
        self.data = {}
        self.load_tensorboard_data()
        
    def load_tensorboard_data(self):
        """Load data from TensorBoard event files"""
        print(f"Loading data from {self.log_dir}")
        
        # Find all event files
        event_files = list(self.log_dir.rglob("events.out.tfevents.*"))
        
        if not event_files:
            print(f"No TensorBoard event files found in {self.log_dir}")
            return
            
        for event_file in event_files:
            try:
                ea = EventAccumulator(str(event_file))
                ea.Reload()
                
                # Get all scalar tags
                tags = ea.Tags()['scalars']
                print(f"Found tags: {tags}")
                
                for tag in tags:
                    scalar_events = ea.Scalars(tag)
                    steps = [event.step for event in scalar_events]
                    values = [event.value for event in scalar_events]
                    
                    if tag not in self.data:
                        self.data[tag] = {'steps': [], 'values': []}
                    
                    self.data[tag]['steps'].extend(steps)
                    self.data[tag]['values'].extend(values)
                    
            except Exception as e:
                print(f"Error loading {event_file}: {e}")
                
        print(f"Loaded {len(self.data)} metrics")
        
    def plot_training_progress(self, save_path="training_visualizations"):
        """Create comprehensive training visualizations"""
        if not self.data:
            print("No data to visualize")
            return
            
        os.makedirs(save_path, exist_ok=True)
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Episode Rewards
        if 'episode/reward' in self.data:
            plt.subplot(3, 3, 1)
            self.plot_metric('episode/reward', 'Episode Reward', 'Episodes', 'Reward', 
                           color='blue', alpha=0.7)
            plt.grid(True, alpha=0.3)
            
        # 2. Average Reward (100 episodes)
        if 'episode/avg_reward_100' in self.data:
            plt.subplot(3, 3, 2)
            self.plot_metric('episode/avg_reward_100', 'Average Reward (100 episodes)', 
                           'Episodes', 'Average Reward', color='green', linewidth=2)
            plt.grid(True, alpha=0.3)
            
        # 3. Best Reward
        if 'episode/best_reward' in self.data:
            plt.subplot(3, 3, 3)
            self.plot_metric('episode/best_reward', 'Best Reward Ever', 
                           'Episodes', 'Best Reward', color='red', linewidth=2)
            plt.grid(True, alpha=0.3)
            
        # 4. Episode Length
        if 'episode/length' in self.data:
            plt.subplot(3, 3, 4)
            self.plot_metric('episode/length', 'Episode Length', 
                           'Episodes', 'Steps', color='orange', alpha=0.7)
            plt.grid(True, alpha=0.3)
            
        # 5. Actor Loss
        actor_loss_key = None
        for key in self.data.keys():
            if 'actor_loss' in key:
                actor_loss_key = key
                break
                
        if actor_loss_key:
            plt.subplot(3, 3, 5)
            self.plot_metric(actor_loss_key, 'Actor Loss', 
                           'Training Steps', 'Loss', color='purple', alpha=0.7)
            plt.grid(True, alpha=0.3)
            
        # 6. Critic Loss
        critic_loss_key = None
        for key in self.data.keys():
            if 'critic_loss' in key:
                critic_loss_key = key
                break
                
        if critic_loss_key:
            plt.subplot(3, 3, 6)
            self.plot_metric(critic_loss_key, 'Critic Loss', 
                           'Training Steps', 'Loss', color='brown', alpha=0.7)
            plt.grid(True, alpha=0.3)
            
        # 7. Entropy Coefficient
        ent_coef_key = None
        for key in self.data.keys():
            if 'ent_coef' in key:
                ent_coef_key = key
                break
                
        if ent_coef_key:
            plt.subplot(3, 3, 7)
            self.plot_metric(ent_coef_key, 'Entropy Coefficient', 
                           'Training Steps', 'Entropy Coef', color='pink', alpha=0.7)
            plt.grid(True, alpha=0.3)
            
        # 8. Learning Rate
        lr_key = None
        for key in self.data.keys():
            if 'learning_rate' in key:
                lr_key = key
                break
                
        if lr_key:
            plt.subplot(3, 3, 8)
            self.plot_metric(lr_key, 'Learning Rate', 
                           'Training Steps', 'Learning Rate', color='gray', alpha=0.7)
            plt.grid(True, alpha=0.3)
            
        # 9. Reward Distribution (histogram)
        if 'episode/reward' in self.data:
            plt.subplot(3, 3, 9)
            rewards = self.data['episode/reward']['values']
            plt.hist(rewards, bins=50, alpha=0.7, color='cyan', edgecolor='black')
            plt.title('Reward Distribution')
            plt.xlabel('Reward')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(f"{save_path}/training_overview.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create separate detailed plots
        self.create_detailed_plots(save_path)
        
    def plot_metric(self, key, title, xlabel, ylabel, color='blue', alpha=1.0, linewidth=1):
        """Plot a single metric"""
        if key not in self.data:
            return
            
        steps = self.data[key]['steps']
        values = self.data[key]['values']
        
        plt.plot(steps, values, color=color, alpha=alpha, linewidth=linewidth)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # Add trend line for reward metrics
        if 'reward' in key and len(values) > 10:
            z = np.polyfit(steps, values, 1)
            p = np.poly1d(z)
            plt.plot(steps, p(steps), "--", color='red', alpha=0.8, linewidth=1)
            
    def create_detailed_plots(self, save_path):
        """Create detailed individual plots"""
        
        # 1. Detailed Reward Analysis
        if 'episode/reward' in self.data:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Raw rewards
            steps = self.data['episode/reward']['steps']
            values = self.data['episode/reward']['values']
            ax1.plot(steps, values, alpha=0.6, color='blue', linewidth=0.8)
            ax1.set_title('Raw Episode Rewards')
            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)
            
            # Moving average
            if len(values) > 50:
                window = min(50, len(values) // 10)
                moving_avg = pd.Series(values).rolling(window=window).mean()
                ax2.plot(steps, moving_avg, color='green', linewidth=2)
                ax2.set_title(f'Moving Average Reward (window={window})')
                ax2.set_xlabel('Episodes')
                ax2.set_ylabel('Average Reward')
                ax2.grid(True, alpha=0.3)
            
            # Reward distribution
            ax3.hist(values, bins=50, alpha=0.7, color='cyan', edgecolor='black')
            ax3.axvline(np.mean(values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(values):.2f}')
            ax3.axvline(np.median(values), color='orange', linestyle='--', 
                       label=f'Median: {np.median(values):.2f}')
            ax3.set_title('Reward Distribution')
            ax3.set_xlabel('Reward')
            ax3.set_ylabel('Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Cumulative reward
            cumulative_reward = np.cumsum(values)
            ax4.plot(steps, cumulative_reward, color='purple', linewidth=2)
            ax4.set_title('Cumulative Reward')
            ax4.set_xlabel('Episodes')
            ax4.set_ylabel('Cumulative Reward')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/detailed_reward_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
            
        # 2. Loss Analysis
        loss_keys = [key for key in self.data.keys() if 'loss' in key]
        if loss_keys:
            fig, axes = plt.subplots(len(loss_keys), 1, figsize=(12, 4*len(loss_keys)))
            if len(loss_keys) == 1:
                axes = [axes]
                
            for i, key in enumerate(loss_keys):
                steps = self.data[key]['steps']
                values = self.data[key]['values']
                axes[i].plot(steps, values, alpha=0.7, linewidth=1)
                axes[i].set_title(f'{key.replace("train/", "").replace("_", " ").title()}')
                axes[i].set_xlabel('Training Steps')
                axes[i].set_ylabel('Loss')
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                if len(values) > 10:
                    z = np.polyfit(steps, values, 1)
                    p = np.poly1d(z)
                    axes[i].plot(steps, p(steps), "--", color='red', alpha=0.8)
                    
            plt.tight_layout()
            plt.savefig(f"{save_path}/loss_analysis.png", dpi=300, bbox_inches='tight')
            plt.show()
            
    def print_training_summary(self):
        """Print a summary of training statistics"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        if 'episode/reward' in self.data:
            rewards = self.data['episode/reward']['values']
            print(f"Total Episodes: {len(rewards)}")
            print(f"Average Reward: {np.mean(rewards):.2f}")
            print(f"Best Reward: {np.max(rewards):.2f}")
            print(f"Worst Reward: {np.min(rewards):.2f}")
            print(f"Reward Std: {np.std(rewards):.2f}")
            
            # Performance categories
            excellent = sum(1 for r in rewards if r > 50)
            good = sum(1 for r in rewards if 0 < r <= 50)
            poor = sum(1 for r in rewards if -50 < r <= 0)
            terrible = sum(1 for r in rewards if r <= -50)
            
            print(f"\nPerformance Distribution:")
            print(f"🟢 Excellent (>50): {excellent} ({excellent/len(rewards)*100:.1f}%)")
            print(f"🟡 Good (0-50): {good} ({good/len(rewards)*100:.1f}%)")
            print(f"🟠 Poor (-50-0): {poor} ({poor/len(rewards)*100:.1f}%)")
            print(f"🔴 Terrible (<-50): {terrible} ({terrible/len(rewards)*100:.1f}%)")
            
        if 'episode/length' in self.data:
            lengths = self.data['episode/length']['values']
            print(f"\nAverage Episode Length: {np.mean(lengths):.1f} steps")
            print(f"Max Episode Length: {np.max(lengths)}")
            print(f"Min Episode Length: {np.min(lengths)}")
            
        # Training convergence analysis
        if 'episode/avg_reward_100' in self.data:
            avg_rewards = self.data['episode/avg_reward_100']['values']
            if len(avg_rewards) >= 100:
                recent_avg = np.mean(avg_rewards[-50:])
                early_avg = np.mean(avg_rewards[:50])
                improvement = recent_avg - early_avg
                print(f"\nConvergence Analysis:")
                print(f"Early Average (first 50): {early_avg:.2f}")
                print(f"Recent Average (last 50): {recent_avg:.2f}")
                print(f"Improvement: {improvement:.2f}")
                
                if improvement > 10:
                    print("✅ Training is converging well!")
                elif improvement > 0:
                    print("🟡 Training is improving slowly")
                else:
                    print("⚠️ Training might be stuck or diverging")
                    
        print("="*60)
        
    def analyze_exploration(self):
        """Analyze exploration behavior"""
        if 'episode/reward' in self.data:
            rewards = self.data['episode/reward']['values']
            
            # Calculate variance over time to see if agent is exploring
            window_size = 50
            if len(rewards) > window_size:
                variances = []
                for i in range(window_size, len(rewards)):
                    window_rewards = rewards[i-window_size:i]
                    variances.append(np.var(window_rewards))
                
                plt.figure(figsize=(12, 6))
                plt.plot(range(window_size, len(rewards)), variances, 
                        alpha=0.7, color='green', linewidth=2)
                plt.title('Reward Variance Over Time (Exploration Indicator)')
                plt.xlabel('Episodes')
                plt.ylabel('Reward Variance')
                plt.grid(True, alpha=0.3)
                
                # High variance = more exploration
                avg_variance = np.mean(variances)
                if avg_variance > 1000:
                    exploration_status = "High Exploration 🔍"
                elif avg_variance > 500:
                    exploration_status = "Moderate Exploration 🔍"
                else:
                    exploration_status = "Low Exploration ⚠️"
                    
                plt.text(0.02, 0.98, f"Status: {exploration_status}", 
                        transform=plt.gca().transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle="round", facecolor='wheat'))
                
                plt.tight_layout()
                plt.savefig("training_visualizations/exploration_analysis.png", dpi=300, bbox_inches='tight')
                plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize SAC training progress')
    parser.add_argument('--log_dir', type=str, default='logs/sac_lunar_lander',
                       help='Path to tensorboard logs directory')
    parser.add_argument('--save_dir', type=str, default='training_visualizations',
                       help='Directory to save visualization plots')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = TrainingVisualizer(args.log_dir)
    
    # Generate visualizations
    print("Creating training visualizations...")
    visualizer.plot_training_progress(args.save_dir)
    
    # Print summary
    visualizer.print_training_summary()
    
    # Analyze exploration
    print("\nAnalyzing exploration behavior...")
    visualizer.analyze_exploration()
    
    print(f"\nAll visualizations saved to: {args.save_dir}")

if __name__ == "__main__":
    main()