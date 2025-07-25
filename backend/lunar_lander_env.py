import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from flask import Flask, request, jsonify
import threading
from waitress import serve
import sys
import io
import queue
import time
import subprocess
import os
import platform
import signal
import atexit

import logging
logging.getLogger('waitress').setLevel(logging.CRITICAL)
logging.getLogger('asyncore').setLevel(logging.CRITICAL)

# Set up logging for this module
logger = logging.getLogger(__name__)

STATE_SIZE = 17
ACTION_SIZE = 6
MAX_EPISODE_STEPS = 4500

# --- TERMINAL REWARD/PENALTY CONSTANTS ---
REWARD_SUCCESS = 100.0
PENALTY_CRASH = -100.0
PENALTY_FAILURE = -50.0

# --- LANDING SUCCESS CRITERIA ---
LANDING_TARGET_RADIUS = 3.0
MAX_LANDING_SPEED_TOTAL = 1.5
MAX_LANDING_TILT_RADIANS = np.deg2rad(12)

# --- SHAPING PENALTY CONSTANTS ---
TILT_PENALTY_THRESHOLD_RADIANS = np.deg2rad(25)

# --- World Bounds ---
BOUNDS_XY = 50000.0
BOUNDS_Z_MAX = 200000.0

class UnrealLunarLanderEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, flask_port=5000, host='0.0.0.0',
             unreal_exe_path=None, launch_unreal=False, ue_launch_args=None,
             max_reset_attempts=3):
        super(UnrealLunarLanderEnv, self).__init__()
        
        self.action_space = Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        
        MAX_ANG_VEL_BOUND = np.deg2rad(720)
        MAX_POSITION_BOUND = BOUNDS_XY / 100.0
        MAX_ALTITUDE_BOUND = BOUNDS_Z_MAX / 100.0
        
        low_bounds = np.array([
            -MAX_POSITION_BOUND, -MAX_POSITION_BOUND, -MAX_ALTITUDE_BOUND, -np.pi, -np.pi, -np.pi,
            -50.0, -50.0, -50.0, -MAX_ANG_VEL_BOUND, -MAX_ANG_VEL_BOUND, -MAX_ANG_VEL_BOUND,
            0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        high_bounds = np.array([
            MAX_POSITION_BOUND, MAX_POSITION_BOUND, MAX_ALTITUDE_BOUND, np.pi, np.pi, np.pi,
            50.0, 50.0, 50.0, MAX_ANG_VEL_BOUND, MAX_ANG_VEL_BOUND, MAX_ANG_VEL_BOUND,
            MAX_ALTITUDE_BOUND, MAX_ALTITUDE_BOUND, MAX_ALTITUDE_BOUND, MAX_ALTITUDE_BOUND, 1.0
        ], dtype=np.float32)
        
        self.observation_space = Box(low=low_bounds, high=high_bounds, shape=(STATE_SIZE,), dtype=np.float32)

        self.flask_app = Flask(__name__)
        self.host = host
        self.port = flask_port
        self.server_thread = None
        self.data_from_unreal = queue.Queue(maxsize=1)
        self.action_for_unreal = queue.Queue(maxsize=1)
        self.unreal_exe_path = unreal_exe_path
        self.launch_unreal = launch_unreal
        self.ue_launch_args = ue_launch_args if ue_launch_args is not None else []
        self.unreal_process = None
        self.current_state = np.zeros(STATE_SIZE, dtype=np.float32)
        self.episode_step_count = 0
        self.is_closed = False
        self.prev_distance_to_target = 0.0
        self.max_reset_attempts = max_reset_attempts
        self.episode_started = False  # Track if episode has actually started
        
        self.last_request_time = 0.0
        # More aggressive throttling to reduce server load
        self.min_request_interval = 0.033  # ~30 FPS instead of 20 FPS
        self.dropped_requests = 0
        self.total_requests = 0

        self._setup_flask_routes()
        self._start_flask_server()
        if self.launch_unreal: 
            self._launch_unreal_engine()
        
        atexit.register(self.close)
        print(f"UnrealLunarLanderEnv initialized. Flask server running on http://{self.host}:{self.port}")

    def _normalize_observation(self, raw_obs):
        raw_obs = np.array(raw_obs, dtype=np.float32)
        normalized_obs = np.zeros_like(raw_obs)
        normalized_obs[0:3] = raw_obs[0:3] / 100.0
        normalized_obs[3:6] = np.deg2rad(raw_obs[3:6])
        normalized_obs[6:9] = raw_obs[6:9] / 100.0
        normalized_obs[9:12] = np.deg2rad(raw_obs[9:12])
        normalized_obs[12:16] = raw_obs[12:16] / 100.0
        normalized_obs[16] = raw_obs[16]
        
        normalized_obs = np.clip(
            normalized_obs, self.observation_space.low, self.observation_space.high
        )
        return normalized_obs.astype(np.float32)

    def _calculate_reward_and_done(self, state):
        reward = 0.0
        done = False
        info = {}

        relative_pos = state[0:3] / 100.0
        orientation = np.deg2rad(state[3:6])
        lin_vel = state[6:9] / 100.0
        hit_flag = state[16]

        distance_to_target = np.linalg.norm(relative_pos[0:2])
        total_speed = np.linalg.norm(lin_vel)
        roll, pitch = orientation[0], orientation[1]

        # Debug logging for first few steps
        if self.episode_step_count < 5:
            logger.info(f"Step {self.episode_step_count}: pos={relative_pos}, hit_flag={hit_flag}, distance={distance_to_target:.2f}")

        # --- TERMINATION LOGIC ---
        # Only consider hit flag if we've had a few steps to actually start the episode
        if hit_flag > 0 and self.episode_step_count > 10:  # Give some buffer time
            done = True
            is_on_target = distance_to_target < LANDING_TARGET_RADIUS
            is_stable_speed = total_speed < MAX_LANDING_SPEED_TOTAL
            is_upright = abs(pitch) < MAX_LANDING_TILT_RADIANS and abs(roll) < MAX_LANDING_TILT_RADIANS

            logger.info(f"Hit detected at step {self.episode_step_count}: on_target={is_on_target}, stable_speed={is_stable_speed}, upright={is_upright}")

            if is_on_target and is_stable_speed and is_upright:
                reward = 100.0
                info['status'] = 'successful_landing'
                logger.info("SUCCESSFUL LANDING!")
            else:
                reward = -10.0
                if is_on_target: reward += 20.0
                if is_stable_speed: reward += 20.0
                if is_upright: reward += 20.0
                info['status'] = 'crash'
                logger.info(f"CRASH: reward={reward}")

            return reward, done, info

        # Check bounds with some buffer and only after episode has started
        if self.episode_step_count > 10:
            if abs(relative_pos[0]) > BOUNDS_XY or abs(relative_pos[1]) > BOUNDS_XY:
                reward = -5.0
                done = True
                info['status'] = 'out_of_bounds'
                logger.info(f"OUT OF BOUNDS at step {self.episode_step_count}: pos={relative_pos}")
                return reward, done, info

        if self.episode_step_count >= MAX_EPISODE_STEPS:
            reward = -1.0
            done = True
            info['status'] = 'max_steps_reached'
            logger.info(f"MAX STEPS REACHED: {self.episode_step_count}")
            return reward, done, info

        # --- CONTINUOUS SHAPING REWARDS ---
        max_per_step = 100.0 / 4500  # ~0.022

        # Progress calculation
        progress = 0.0
        if hasattr(self, 'prev_distance_to_target'):
            progress = self.prev_distance_to_target - distance_to_target
        self.prev_distance_to_target = distance_to_target

        # Determine sign: +1 if moving toward target, -1 if moving away
        direction_sign = 1.0 if progress > 0 else -1.0

        # Equally weighted components (will normalize by num_conditions=5)
        components = []

        # 1. Distance closeness
        max_distance = np.sqrt(BOUNDS_XY**2 + BOUNDS_XY**2)
        closeness = (max_distance - distance_to_target) / max_distance
        components.append(closeness * direction_sign)

        # 2. Height - closer to ground is better if moving toward target
        height = abs(relative_pos[2])
        height_score = (50.0 - height) / 50.0 if height < 50.0 else 0.0
        components.append(height_score * direction_sign)

        # 3. Upright orientation
        tilt_penalty = (abs(roll) + abs(pitch)) / (2 * np.pi)
        uprightness = (1.0 - tilt_penalty)
        components.append(uprightness * direction_sign)

        # 4. Controlled speed
        if total_speed < 2.0:
            speed_score = 1.0
        elif total_speed < 3.5:
            speed_score = 0.7
        else:
            speed_score = -0.5  # actively penalize very high speeds
        components.append(speed_score * direction_sign)

        # 5. Proximity bonus if very close to target
        proximity_score = (20.0 - distance_to_target) / 20.0 if distance_to_target < 20.0 else 0.0
        components.append(proximity_score * direction_sign)

        # --- Average them so sum stays ~0.0 to 1.0 ---
        step_reward = sum(components) / len(components) * max_per_step

        # Also add a small time penalty to encourage quicker episodes
        step_reward -= 0.001

        reward += np.clip(step_reward, -max_per_step, max_per_step)

        # Log reward occasionally
        if self.episode_step_count % 100 == 0:
            logger.info(f"Step {self.episode_step_count}: step_reward={step_reward:.4f}, total_reward={reward:.4f}")

        return reward, done, info

    def reset(self, seed=None, options=None): # (FIXED) Updated method signature
        if self.is_closed: 
            raise RuntimeError("Cannot reset a closed environment")
        if seed is not None: 
            np.random.seed(seed)
        
        # (FIXED) Extract initial_params from the standard 'options' dictionary
        initial_params = options.get('initial_params') if options else None

        self.episode_started = False
        self.episode_step_count = 0
        
        for attempt in range(self.max_reset_attempts):
            try:
                # Pass the extracted params to the attempt
                result = self._attempt_reset(attempt, initial_params)
                if result:
                    self.episode_started = True
                    logger.info(f"Reset successful on attempt {attempt + 1}")
                    return result
            except Exception as e:
                logger.warning(f"Reset attempt {attempt + 1} failed: {e}")
                if attempt == self.max_reset_attempts - 1:
                    logger.error("All reset attempts failed. Returning zero state.")
                    self.current_state = np.zeros(STATE_SIZE, dtype=np.float32)
                    return self._normalize_observation(self.current_state), {}
                time.sleep(1)
        
        self.current_state = np.zeros(STATE_SIZE, dtype=np.float32)
        return self._normalize_observation(self.current_state), {}

    def _attempt_reset(self, attempt_num, initial_params=None):
        # Clear queues
        while not self.data_from_unreal.empty(): 
            try: self.data_from_unreal.get_nowait()
            except queue.Empty: break
        while not self.action_for_unreal.empty(): 
            try: self.action_for_unreal.get_nowait()
            except queue.Empty: break
                
        if self.launch_unreal and (not self.unreal_process or self.unreal_process.poll() is not None):
            self._launch_unreal_engine()
            time.sleep(3)
        
        # (FIXED) Refactored parameter handling to be more robust
        # This removes the if/else block that was causing the issue.
        params = initial_params or {} # Ensure params is a dictionary

        start_params = {
            "latitude": params.get('latitude', 0.0),
            "longitude": params.get('longitude', 0.0), 
            "height": params.get('height', 100.0),
            "linear_velocity": [
                params.get('vx', 0.0),
                params.get('vy', 0.0),
                params.get('vz', -1.0)
            ],
            "orientation": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0]
        }
    
        reset_command = {"command": "reset", "params": start_params}
        
        try:
            self.action_for_unreal.put(reset_command, timeout=5.0)
        except queue.Full:
            while not self.action_for_unreal.empty():
                try: self.action_for_unreal.get_nowait()
                except queue.Empty: break
            self.action_for_unreal.put(reset_command, timeout=5.0)
            
        print(f"Gym env: RESET attempt {attempt_num + 1}. Waiting for UE...")
        
        # Wait for UE response with timeout
        timeout_duration = 30.0  # Reduced from 120s
        start_time = time.time()
        while (time.time() - start_time) < timeout_duration:
            try:
                ue_data = self.data_from_unreal.get(timeout=1.0)
                if ue_data.get("start") is True and "state" in ue_data:
                    self.action_for_unreal.put({"command": "start_confirmed"})
                    initial_state_from_ue = np.array(ue_data["state"], dtype=np.float32)
                    print(f"Gym env: 'start' signal received on attempt {attempt_num + 1}.")
                    
                    self.current_state = initial_state_from_ue
                    pos_m = self.current_state[0:3] / 100.0
                    self.prev_distance_to_target = np.linalg.norm(pos_m[0:2])
                    
                    return self._normalize_observation(initial_state_from_ue), {}
                else:
                    self.action_for_unreal.put({"command": "dummy"})
            except queue.Empty: 
                continue
            except Exception as e:
                print(f"[Warning] Exception during reset attempt {attempt_num + 1}: {e}")
                continue

        # If we reach here, this attempt timed out
        raise TimeoutError(f"Reset attempt {attempt_num + 1} timed out after {timeout_duration}s")

    def step(self, action_payload):
        if self.is_closed: 
            raise RuntimeError("Cannot step in a closed environment")
            
        self.episode_step_count += 1
        step_command = {"command": "step", "action": [float(a) for a in action_payload]}

        try:
            self.action_for_unreal.put(step_command, timeout=2.0)
        except queue.Full:
            print("[Warning] Action queue full during step")
            reward, _, info = self._calculate_reward_and_done(self.current_state)
            normalized_state = self._normalize_observation(self.current_state)
            return normalized_state, reward + PENALTY_FAILURE, True, False, {"error": "Action queue full", **info}
            
        try:
            ue_data = self.data_from_unreal.get(timeout=10.0)  # Reduced timeout
            new_state_from_ue = np.array(ue_data["state"], dtype=np.float32)
            
            reward, done, info = self._calculate_reward_and_done(new_state_from_ue)
            self.current_state = new_state_from_ue
            
            normalized_state = self._normalize_observation(new_state_from_ue)
            truncated = info.get('status') == 'max_steps_reached'
            return normalized_state, reward, done, truncated, info
        except Exception as e:
            print(f"[Warning] UE response timeout during step: {e}")
            # Instead of ending episode, return current state with penalty
            reward, _, info = self._calculate_reward_and_done(self.current_state)
            normalized_state = self._normalize_observation(self.current_state)
            # Don't terminate - let the training continue with penalty
            return normalized_state, reward - 10.0, False, False, {"error": f"UE timeout: {e}", **info}

    def close(self):
        if self.is_closed: return
        self.is_closed = True
        print("[Info] Closing UnrealLunarLanderEnv...")
        if self.unreal_process and self.unreal_process.poll() is None:
            pid = self.unreal_process.pid
            print(f"[Info] Terminating UE process tree (PID: {pid})...")
            try:
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/F", "/PID", str(pid), "/T"], check=True, capture_output=True, timeout=10)
                else:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                self.unreal_process.wait(timeout=5)
            except Exception as e:
                print(f"[Warning] Error during UE process cleanup: {e}")
                try: self.unreal_process.kill()
                except: pass
            self.unreal_process = None
            print("[Info] UE process terminated.")
        print("[Info] UnrealLunarLanderEnv closed successfully.")

    def render(self, mode='human'):
        pass
        
    def _launch_unreal_engine(self):
        if not self.unreal_exe_path or not os.path.exists(self.unreal_exe_path): 
            print(f"[Warning] Unreal Engine executable not found at: {self.unreal_exe_path}")
            return
        try:
            cmd = [self.unreal_exe_path] + self.ue_launch_args
            self.unreal_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == "Windows" else 0
            )
            print(f"UE process started (PID: {self.unreal_process.pid}).")
            time.sleep(2)
        except Exception as e:
            self.unreal_process = None; print(f"[Error] Failed to launch UE: {e}")

    def _setup_flask_routes(self):
        @self.flask_app.route('/control', methods=['POST'])
        def control_lander_route():
            if self.is_closed: 
                return jsonify({"command": "dummy", "error": "Environment is closed"}), 500
            
            current_time = time.time()
            self.total_requests += 1
            
            try:
                data = request.json
                
                # Check if this is a critical request that should NEVER be dropped
                is_critical_request = False
                
                # Check for start signal
                if data.get("start") is True:
                    is_critical_request = True
                    print("[Flask] Critical: Start signal received - not dropping")
                
                # Check for hit flag (landing/crash event) - last value in state array
                elif "state" in data and len(data["state"]) >= 17:
                    hit_flag = data["state"][16]  # Last element is hit flag
                    if hit_flag > 0:
                        is_critical_request = True
                        print(f"[Flask] Critical: Hit event detected (hit_flag={hit_flag}) - not dropping")
                
                # Check if we should throttle (only for non-critical requests after environment has started)
                should_throttle = (
                    not is_critical_request and 
                    hasattr(self, 'episode_step_count') and 
                    self.episode_step_count > 0 and  # Only throttle after first episode has started
                    current_time - self.last_request_time < self.min_request_interval
                )
                
                if should_throttle:
                    self.dropped_requests += 1
                    # Log throttling stats every 100 dropped requests
                    if self.dropped_requests % 100 == 0:
                        drop_rate = (self.dropped_requests / self.total_requests) * 100
                        print(f"[Throttling] Dropped {self.dropped_requests} requests ({drop_rate:.1f}% drop rate)")
                    
                    # Return the last action or dummy action quickly
                    try:
                        # Try to get existing action without blocking
                        action = self.action_for_unreal.get_nowait()
                        return jsonify(action)
                    except queue.Empty:
                        return jsonify({"command": "dummy"})
                
                # Update last request time only for non-dropped requests
                self.last_request_time = current_time
                
                # Process the request (critical requests or non-throttled requests)
                # Non-blocking queue management
                if not self.data_from_unreal.full():
                    self.data_from_unreal.put_nowait(data)
                else:
                    # For critical requests, we must ensure they get through
                    if is_critical_request:
                        # Force clear queue to make room for critical data
                        try: 
                            old_data = self.data_from_unreal.get_nowait()
                            print(f"[Flask] Cleared queue for critical request (removed non-critical data)")
                        except queue.Empty: 
                            pass
                        self.data_from_unreal.put_nowait(data)
                    else:
                        # For non-critical, replace oldest data as before
                        try: 
                            self.data_from_unreal.get_nowait()
                        except queue.Empty: 
                            pass
                        self.data_from_unreal.put_nowait(data)
                
                # Get action with timeout
                return jsonify(self.action_for_unreal.get(timeout=5.0))
                
            except queue.Empty: 
                return jsonify({"command": "dummy", "error": "No command available"}), 200
            except Exception as e: 
                print(f"[Flask] Error processing request: {e}")
                return jsonify({"command": "dummy", "error": str(e)}), 200

    
    def _start_flask_server(self):
        if self.server_thread and self.server_thread.is_alive():
            return
        self.server_thread = threading.Thread(
            target=lambda: serve(
                self.flask_app,
                host=self.host,
                port=self.port,
                threads=4,  # Reduced from 32
                connection_limit=100,  # Reduced from 1000
                cleanup_interval=10,  # Reduced from 30
                channel_timeout=30,  # Reduced from 120
                log_socket_errors=False,
                recv_bytes=8192,  # Limit receive buffer
                send_bytes=8192   # Limit send buffer
            )
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        time.sleep(1)


gym.register(
    id='UnrealLunarLander-v0',
    entry_point=UnrealLunarLanderEnv,
    max_episode_steps=MAX_EPISODE_STEPS,
)