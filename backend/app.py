from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import time
import threading
import numpy as np
import psutil
import subprocess
from stable_baselines3 import SAC
from lunar_lander_env import UnrealLunarLanderEnv
import torch
import signal
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class ActionAdapter:
    """Adapter to convert 3D action to 6D action by adding zeros"""
    def __init__(self, env):
        self.env = env
        
    def action(self, action):
        # Convert 3D action to 6D by appending zeros
        extended_action = np.zeros(6, dtype=np.float32)
        extended_action[:3] = action
        return extended_action

class LunarLanderSimulation:
    def __init__(self):
        self.model = None
        self.env = None
        self.is_running = False
        self.stop_requested = False  # Add explicit stop flag
        self.current_episode_reward = 0.0
        self.episode_done = False
        self.simulation_thread = None
        self.model_path = None
        self.unreal_process = None
        self.step_count = 0
        self.episode_start_time = None
        self.thread_lock = threading.Lock()  # Add thread synchronization
        
    def load_model(self, model_path):
        """Load the trained SAC model"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = SAC.load(model_path)
            self.model_path = model_path
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def find_unreal_executable(self):
        """Try to find Unreal Engine executable"""
        possible_paths = [
            "C:/Users/jemim/VSCode Projects/Lunar Lander UI/Unreal Engine/lunar_lander.exe"
            # Add more possible paths here
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def kill_existing_unreal_processes(self):
        """Kill any existing Unreal Engine processes"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if 'unreal' in proc.info['name'].lower() or 'ue' in proc.info['name'].lower() or 'lunar_lander' in proc.info['name'].lower():
                    logger.info(f"Killing existing Unreal process: {proc.info['name']} (PID: {proc.info['pid']})")
                    try:
                        proc.kill()
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error killing existing processes: {e}")
    
    def start_simulation(self, location, linear_velocity):
        """Start the simulation with given parameters"""
        with self.thread_lock:
            if self.is_running:
                return {"error": "Simulation already running"}
            
            if not self.model:
                return {"error": "No model loaded"}
            
            # Reset stop flag when starting new simulation
            self.stop_requested = False
        
        try:
            # Clean up any existing processes
            self.kill_existing_unreal_processes()
            time.sleep(2)
            
            # Find Unreal executable
            unreal_exe = self.find_unreal_executable()
            if not unreal_exe:
                return {"error": "Unreal Engine executable not found"}
            
            # Create environment
            env_kwargs = {
                'flask_port': 5000,
                'host': '0.0.0.0',
                'unreal_exe_path': unreal_exe,
                'launch_unreal': True,
                'ue_launch_args': ['-game', '-ResX=800', '-ResY=600', '-windowed'],
                'max_reset_attempts': 3
            }
            
            self.env = UnrealLunarLanderEnv(**env_kwargs)
            
            # Start simulation in a separate thread
            self.simulation_thread = threading.Thread(
                target=self._run_simulation,
                args=(location, linear_velocity),
                daemon=True
            )
            self.simulation_thread.start()
            
            return {"status": "Simulation started", "message": "Launching Unreal Engine..."}
            
        except Exception as e:
            logger.error(f"Error starting simulation: {e}")
            return {"error": f"Failed to start simulation: {str(e)}"}
    
    # In app.py, inside the LunarLanderSimulation class...

    def _run_simulation(self, location, linear_velocity):
        """Run the simulation episode"""
        try:
            with self.thread_lock:
                self.is_running = True
                self.current_episode_reward = 0.0
                self.episode_done = False
                self.step_count = 0
                self.episode_start_time = time.time()
            
            logger.info(f"Starting episode with location: {location}, velocity: {linear_velocity}")
            
            if self.stop_requested:
                logger.info("Stop requested before simulation start, aborting")
                return
            
            # (FIXED) Correctly map UI labels to parameters
            initial_params = {
                'latitude': location['y'],    # Latitude from UI
                'longitude': location['x'],   # Longitude from UI
                'height': location['z'],
                'vx': linear_velocity['x'],
                'vy': linear_velocity['y'],
                'vz': linear_velocity['z']
            }

            # (FIXED) Pass parameters through the 'options' dictionary
            observation, _ = self.env.reset(options={'initial_params': initial_params})
            logger.info("Environment reset successful, starting episode loop...")
            
            # ... rest of the method remains the same
            
            max_steps = 4500  # Match the environment's MAX_EPISODE_STEPS
            
            while not self.episode_done and self.step_count < max_steps and self.is_running and not self.stop_requested:
                try:
                    # Check stop condition at beginning of each step
                    if self.stop_requested:
                        logger.info("Stop requested during simulation, breaking loop")
                        break
                    
                    # Get action from model
                    action, _ = self.model.predict(observation, deterministic=True)
                    
                    # Convert 3D action to 6D
                    extended_action = np.zeros(6, dtype=np.float32)
                    extended_action[:3] = action
                    
                    # Take step in environment
                    observation, reward, done, truncated, info = self.env.step(extended_action)
                    
                    with self.thread_lock:
                        self.current_episode_reward += reward
                        self.step_count += 1
                    
                    # Log progress every 100 steps
                    if self.step_count % 100 == 0:
                        logger.info(f"Step {self.step_count}: reward={reward:.3f}, total_reward={self.current_episode_reward:.2f}")
                    
                    # Check termination conditions
                    if done or truncated:
                        with self.thread_lock:
                            self.episode_done = True
                        episode_duration = time.time() - self.episode_start_time
                        logger.info(f"Episode completed after {self.step_count} steps ({episode_duration:.1f}s)")
                        logger.info(f"Final reward: {self.current_episode_reward:.2f}")
                        logger.info(f"Episode status: {info.get('status', 'unknown')}")
                        break
                    
                    # Add a small delay to prevent overwhelming the system
                    time.sleep(0.01)
                    
                except Exception as step_error:
                    logger.error(f"Error during step {self.step_count}: {step_error}")
                    # Check if we should continue or stop
                    if self.stop_requested:
                        break
                    # Don't break immediately on step errors, try to continue
                    continue
            
            # Check why the episode ended
            if self.stop_requested:
                logger.info("Episode terminated due to stop request")
            elif self.step_count >= max_steps:
                logger.warning(f"Episode terminated due to max steps ({max_steps})")
            elif not self.is_running:
                logger.info("Episode terminated due to manual stop")
            
        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            with self.thread_lock:
                self.current_episode_reward = -1000  # Indicate error
                self.episode_done = True
        
        finally:
            # Clean up
            self._cleanup_simulation()
    
    def _cleanup_simulation(self):
        """Clean up after simulation"""
        try:
            logger.info("Starting simulation cleanup...")
            
            if self.env:
                try:
                    self.env.close()
                except Exception as e:
                    logger.error(f"Error closing environment: {e}")
                finally:
                    self.env = None
                    logger.info("Environment closed")
            
            # Kill Unreal process after a short delay
            time.sleep(2)
            self.kill_existing_unreal_processes()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        finally:
            with self.thread_lock:
                self.is_running = False
                self.stop_requested = False  # Reset stop flag
            logger.info("Simulation cleanup complete")
    
    def get_status(self):
        """Get current simulation status"""
        with self.thread_lock:
            return {
                "is_running": self.is_running,
                "episode_done": self.episode_done,
                "current_reward": self.current_episode_reward,
                "model_loaded": self.model is not None,
                "model_path": self.model_path,
                "step_count": self.step_count,
                "episode_duration": time.time() - self.episode_start_time if self.episode_start_time else 0,
                "stop_requested": self.stop_requested
            }
    
    def stop_simulation(self):
        """Stop the current simulation"""
        with self.thread_lock:
            if self.is_running:
                self.stop_requested = True
                self.episode_done = True
                logger.info("Simulation stop requested")
                
                # Wait for the simulation thread to finish (with timeout)
                if self.simulation_thread and self.simulation_thread.is_alive():
                    logger.info("Waiting for simulation thread to finish...")
                    # Don't wait too long, force cleanup if needed
                    self.simulation_thread.join(timeout=5.0)
                    if self.simulation_thread.is_alive():
                        logger.warning("Simulation thread did not finish cleanly")
                
                # Force cleanup if thread is still running
                if self.is_running:
                    self._cleanup_simulation()
                
                return {"status": "Simulation stopped"}
            else:
                return {"status": "No simulation running"}

# Global simulation instance
sim = LunarLanderSimulation()

@app.route('/api/load_model', methods=['POST'])
def load_model():
    """Load a trained model"""
    try:
        data = request.json
        model_path = data.get('model_path')
        
        if not model_path:
            return jsonify({"error": "No model path provided"}), 400
        
        success = sim.load_model(model_path)
        if success:
            return jsonify({"status": "Model loaded successfully", "model_path": model_path})
        else:
            return jsonify({"error": "Failed to load model"}), 400
            
    except Exception as e:
        logger.error(f"Error in load_model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/launch', methods=['POST'])
def launch_simulation():
    """Launch simulation with given parameters"""
    try:
        data = request.json
        location = data.get('location')
        linear_velocity = data.get('linearVelocity')
        
        if not location or not linear_velocity:
            return jsonify({"error": "Missing location or linearVelocity parameters"}), 400
        
        result = sim.start_simulation(location, linear_velocity)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in launch_simulation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get simulation status"""
    try:
        return jsonify(sim.get_status())
    except Exception as e:
        logger.error(f"Error in get_status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop current simulation"""
    try:
        result = sim.stop_simulation()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in stop_simulation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available trained models"""
    try:
        model_dir = "C:/Users/jemim/VSCode Projects/Lunar Lander UI/models/sac_lunar_lander"
        models = []
        
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.zip'):
                    file_path = os.path.join(model_dir, file)
                    file_size = os.path.getsize(file_path)
                    modified_time = os.path.getmtime(file_path)
                    
                    models.append({
                        "name": file,
                        "path": file_path,
                        "size": file_size,
                        "modified": modified_time
                    })
        
        # Sort by modification time (newest first)
        models.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({"models": models})
        
    except Exception as e:
        logger.error(f"Error in list_models: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Lunar Lander Backend is running",
        "model_loaded": sim.model is not None,
        "simulation_running": sim.is_running
    })

def find_latest_model():
    """Find the latest trained model"""
    model_dir = "C:/Users/jemim/VSCode Projects/Lunar Lander UI/models/sac_lunar_lander"
    if not os.path.exists(model_dir):
        return None
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
    if not model_files:
        return None
    
    # Sort by modification time to get the latest
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    return os.path.join(model_dir, model_files[0])

def signal_handler(signum, frame):
    """Handle graceful shutdown"""
    logger.info(f"Received signal {signum}, shutting down...")
    sim.stop_simulation()
    sys.exit(0)

if __name__ == '__main__':
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Try to load the latest model automatically
    latest_model = find_latest_model()
    if latest_model:
        logger.info(f"Auto-loading latest model: {latest_model}")
        sim.load_model(latest_model)
    else:
        logger.info("No trained models found. Please train a model first or load one manually.")
    
    logger.info("Starting Lunar Lander Backend Server...")
    logger.info("Backend will be available at http://localhost:5010")
    logger.info("Use Ctrl+C to stop the server")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5010,
        debug=False,  # Set to True for development
        threaded=True
    )