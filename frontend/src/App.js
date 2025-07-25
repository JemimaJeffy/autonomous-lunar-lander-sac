import React, { useState, useRef, useEffect } from 'react';
import Spline from '@splinetool/react-spline';

// API Configuration
const API_BASE_URL = 'http://localhost:5010/api';

// TextLabel component
const TextLabel = ({ topText, bottomText, topTextSize = '16px', bottomTextSize = '32px' }) => {
  return (
    <div style={{
      position: 'absolute',
      top: '33.33%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      zIndex: 2,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '8px',
      color: '#E0E0E0',
      textAlign: 'center',
      userSelect: 'none'
    }}>
      {topText && (
        <div style={{
          fontSize: topTextSize,
          fontWeight: '400',
          letterSpacing: '1px',
          textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)'
        }}>
          {topText}
        </div>
      )}
      {bottomText && (
        <div style={{
          fontSize: bottomTextSize,
          fontWeight: '600',
          letterSpacing: '1px',
          textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)'
        }}>
          {bottomText}
        </div>
      )}
    </div>
  );
};

// InputSlider component
const InputSlider = ({ label, value, onChange, min = 0, max = 100, orientation = 'horizontal' }) => {
  const handleSliderChange = (e) => {
    onChange(parseFloat(e.target.value));
  };

  const handleInputChange = (e) => {
    const newValue = parseFloat(e.target.value);
    if (!isNaN(newValue) && newValue >= min && newValue <= max) {
      onChange(newValue);
    }
  };

  if (orientation === 'vertical') {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '10px',
        height: '90%',
        justifyContent: 'flex-start',
        paddingTop: '0px'
      }}>
        <div style={{
          height: '180px',
          width: '20px',
          position: 'relative',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <input
            type="range"
            min={min}
            max={max}
            step="0.1"
            value={value}
            onChange={handleSliderChange}
            style={{
              width: '180px',
              height: '6px',
              background: 'linear-gradient(to right, #2979FF, #666)',
              borderRadius: '3px',
              outline: 'none',
              cursor: 'pointer',
              transform: 'rotate(-90deg)',
              transformOrigin: 'center',
              appearance: 'none',
              WebkitAppearance: 'none'
            }}
          />
        </div>
        <input
          type="number"
          value={value.toFixed(1)}
          onChange={handleInputChange}
          min={min}
          max={max}
          step="0.1"
          style={{
            width: '60px',
            padding: '4px 8px',
            background: '#333',
            border: '1px solid #555',
            borderRadius: '4px',
            color: '#E0E0E0',
            fontSize: '12px',
            textAlign: 'center'
          }}
        />
        <label style={{
          color: '#E0E0E0',
          fontSize: '14px',
          fontWeight: '500',
          textAlign: 'center'
        }}>
          {label}
        </label>
      </div>
    );
  }

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '15px',
      width: '100%',
      marginBottom: '32px'
    }}>
      <label style={{
        color: '#E0E0E0',
        fontSize: '14px',
        fontWeight: '500',
        minWidth: '80px',
        textAlign: 'left'
      }}>
        {label}
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step="0.1"
        value={value}
        onChange={handleSliderChange}
        style={{
          flex: 1,
          height: '6px',
          background: 'linear-gradient(to right, #2979FF, #666)',
          borderRadius: '3px',
          outline: 'none',
          cursor: 'pointer',
          appearance: 'none',
          WebkitAppearance: 'none'
        }}
      />
      <input
        type="number"
        value={value.toFixed(1)}
        onChange={handleInputChange}
        min={min}
        max={max}
        step="0.1"
        style={{
          width: '80px',
          padding: '4px 8px',
          background: '#333',
          border: '1px solid #555',
          borderRadius: '4px',
          color: '#E0E0E0',
          fontSize: '12px',
          textAlign: 'center'
        }}
      />
    </div>
  );
};

// Status Display Component
const StatusDisplay = ({ status, isLoading }) => {
  if (isLoading) {
    return (
      <div style={{
        color: '#FFD700',
        fontSize: '14px',
        textAlign: 'center',
        marginBottom: '10px'
      }}>
        Loading...
      </div>
    );
  }

  if (!status) return null;

  return (
    <div style={{
      background: 'rgba(30, 30, 30, 0.8)',
      border: '1px solid rgba(255, 255, 255, 0.2)',
      borderRadius: '6px',
      padding: '15px',
      marginBottom: '20px',
      color: '#E0E0E0',
      fontSize: '14px'
    }}>
      <div>Status: {status.is_running ? 'Running' : 'Stopped'}</div>
      <div>Model Loaded: {status.model_loaded ? 'Yes' : 'No'}</div>
      {status.is_running && (
        <div>Current Reward: {status.current_reward?.toFixed(2) || 'N/A'}</div>
      )}
      {status.episode_done && (
        <div style={{ color: '#4CAF50' }}>Episode Complete!</div>
      )}
    </div>
  );
};

function App() {
    const [location, setLocation] = useState({ x: 0, y: 0, z: 125 });
    const [linearVelocity, setLinearVelocity] = useState({ x: 0, y: 0, z: -0.75 });
    const [labelText, setLabelText] = useState({ top: '', bottom: '' });
    const [isLaunching, setIsLaunching] = useState(false);
    const [simulationStatus, setSimulationStatus] = useState(null);
    const [statusLoading, setStatusLoading] = useState(false);
    const [backendConnected, setBackendConnected] = useState(false);
    
    const splineRef = useRef(null);
    const mousePos = useRef({ x: 0, y: 0 });
    const lastMousePos = useRef({ x: 0, y: 0 });
    const statusInterval = useRef(null);

    // API Functions
    const checkBackendHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        setBackendConnected(true);
        return data;
      } catch (error) {
        console.error('Backend health check failed:', error);
        setBackendConnected(false);
        return null;
      }
    };

    const getSimulationStatus = async () => {
      try {
        setStatusLoading(true);
        const response = await fetch(`${API_BASE_URL}/status`);
        const data = await response.json();
        setSimulationStatus(data);
        return data;
      } catch (error) {
        console.error('Error getting simulation status:', error);
        return null;
      } finally {
        setStatusLoading(false);
      }
    };

    const launchSimulation = async (location, linearVelocity) => {
      try {
        const response = await fetch(`${API_BASE_URL}/launch`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            location,
            linearVelocity
          })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
          throw new Error(data.error || 'Launch failed');
        }
        
        return data;
      } catch (error) {
        console.error('Error launching simulation:', error);
        throw error;
      }
    };

    const stopSimulation = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/stop`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          }
        });
        
        const data = await response.json();
        return data;
      } catch (error) {
        console.error('Error stopping simulation:', error);
        throw error;
      }
    };

    // Check backend health on component mount
    useEffect(() => {
      checkBackendHealth().then(health => {
        if (health) {
          console.log('Backend connected:', health);
          getSimulationStatus();
        } else {
          setLabelText({ 
            top: 'Backend Disconnected', 
            bottom: 'Please start the Python backend' 
          });
        }
      });
    }, []);

    // Set up status polling when simulation is running
    useEffect(() => {
      if (simulationStatus?.is_running) {
        statusInterval.current = setInterval(() => {
          getSimulationStatus();
        }, 2000); // Poll every 2 seconds
      } else if (statusInterval.current) {
        clearInterval(statusInterval.current);
        statusInterval.current = null;
      }

      return () => {
        if (statusInterval.current) {
          clearInterval(statusInterval.current);
        }
      };
    }, [simulationStatus?.is_running]);

    // Update label text based on simulation status
    useEffect(() => {
      if (!backendConnected) {
        setLabelText({ 
          top: 'Backend Disconnected', 
          bottom: 'Start Python backend on port 5010' 
        });
      } else if (simulationStatus?.is_running) {
        setLabelText({ 
          top: 'Simulation Running', 
          bottom: `Reward: ${simulationStatus.current_reward?.toFixed(2) || 'N/A'}` 
        });
      } else if (simulationStatus?.episode_done) {
        setLabelText({ 
          top: 'Episode Complete', 
          bottom: `Final Reward: ${simulationStatus.current_reward?.toFixed(2) || 'N/A'}` 
        });
      } else if (simulationStatus?.model_loaded) {
        setLabelText({ 
          top: 'Ready to Launch', 
          bottom: 'Model Loaded Successfully' 
        });
      } else {
        setLabelText({ 
          top: 'No Model Loaded', 
          bottom: 'Please load a trained model' 
        });
      }
    }, [backendConnected, simulationStatus]);

    useEffect(() => {
        const handleMouseMove = (e) => {
            const deltaX = e.clientX - lastMousePos.current.x;
            const deltaY = e.clientY - lastMousePos.current.y;
            
            lastMousePos.current = { x: e.clientX, y: e.clientY };
            
            if (splineRef.current && splineRef.current.findObjectByName) {
                try {
                    const moon = splineRef.current.findObjectByName('Moon') || 
                                splineRef.current.findObjectByName('Sphere') ||
                                splineRef.current.scene?.children?.[0];
                    
                    if (moon && moon.rotation) {
                        // Subtle rotation based on mouse movement
                        moon.rotation.y += deltaX * 0.001;
                        moon.rotation.x += deltaY * 0.001;
                    }
                } catch (error) {
                    console.log('Moon object not found or not ready');
                }
            }
        };

        const handleWheel = (e) => {
            e.preventDefault();
            
            if (splineRef.current && splineRef.current.findObjectByName) {
                try {
                    const moon = splineRef.current.findObjectByName('Moon') || 
                                splineRef.current.findObjectByName('Sphere') ||
                                splineRef.current.scene?.children?.[0];
                    
                    if (moon && moon.position) {
                        // Zoom in/out based on scroll
                        const zoomSpeed = 0.1;
                        moon.position.z += e.deltaY * zoomSpeed;
                        
                        // Clamp the zoom to reasonable bounds
                        moon.position.z = Math.max(-50, Math.min(50, moon.position.z));
                    }
                } catch (error) {
                    console.log('Moon object not found or not ready');
                }
            }
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('wheel', handleWheel, { passive: false });

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('wheel', handleWheel);
        };
    }, []);

    const handleLaunch = async () => {
        if (!backendConnected) {
            alert('Backend not connected. Please start the Python backend server.');
            return;
        }

        if (!simulationStatus?.model_loaded) {
            alert('No model loaded. Please load a trained model first.');
            return;
        }

        if (isLaunching || simulationStatus?.is_running) {
            alert('Simulation is already running.');
            return;
        }

        setIsLaunching(true);
        
        try {
            const result = await launchSimulation(location, linearVelocity);
            console.log('Launch successful:', result);
            
            // Start polling for status updates
            setTimeout(() => {
                getSimulationStatus();
            }, 1000);
            
        } catch (error) {
            console.error('Launch failed:', error);
            alert(`Launch failed: ${error.message}`);
        } finally {
            setIsLaunching(false);
        }
    };

    const handleStop = async () => {
        if (!simulationStatus?.is_running) {
            alert('No simulation is currently running.');
            return;
        }

        try {
            await stopSimulation();
            console.log('Simulation stopped');
            getSimulationStatus();
        } catch (error) {
            console.error('Error stopping simulation:', error);
            alert(`Error stopping simulation: ${error.message}`);
        }
    };

    const onSplineLoad = (spline) => {
        splineRef.current = spline;
        console.log('Spline scene loaded');
    };

    return (
        <div style={{
            position: 'relative',
            width: '100vw',
            height: '100vh',
            overflow: 'hidden'
        }}>
            {/* Spline Background */}
            <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: -1
            }}>
                <Spline 
                    scene="https://prod.spline.design/2ufhELZZbvKUflUX/scene.splinecode"
                    onLoad={onSplineLoad}
                />
            </div>

            {/* Connection Status Indicator */}
            <div style={{
                position: 'absolute',
                top: '20px',
                right: '20px',
                zIndex: 3,
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                color: '#E0E0E0',
                fontSize: '12px'
            }}>
                <div 
                    style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: backendConnected ? '#4CAF50' : '#F44336'
                    }}
                />
                Backend {backendConnected ? 'Connected' : 'Disconnected'}
            </div>

            {/* Text Label */}
            <TextLabel 
                topText={labelText.top}
                bottomText={labelText.bottom}
                topTextSize="16px"
                bottomTextSize="32px"
            />

            {/* Main Content */}
            <div style={{
                position: 'relative',
                zIndex: 1,
                display: 'flex',
                flexDirection: 'column',
                gap: '32px',
                width: '100%',
                height: '100%',
                alignItems: 'center',
                justifyContent: 'flex-end',
                padding: '2rem',
                paddingBottom: '3rem',
                boxSizing: 'border-box'
            }}>
                <div style={{
                    display: 'flex',
                    gap: '80px',
                    alignItems: 'stretch',
                    width: '100%',
                    maxWidth: '75vw',
                    height: '33vh',
                    justifyContent: 'center'
                }}>
                    {/* Horizontal Card - Location */}
                    <div style={{
                        width: '50vw',
                        background: 'rgba(30, 30, 30, 0.3)',
                        backdropFilter: 'blur(20px)',
                        WebkitBackdropFilter: 'blur(20px)',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        borderRadius: '8px',
                        padding: '40px',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'center',
                        gap: '24px',
                        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                    }}>
                        <div style={{
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'center',
                            height: '100%'
                        }}>
                            <h3 style={{
                                margin: '0 0 20px 0',
                                textAlign: 'left',
                                fontWeight: '500',
                                color: '#E0E0E0'
                            }}>Location</h3>
                            <InputSlider 
                                label="Longitude" 
                                value={location.x} 
                                onChange={val => setLocation({ ...location, x: val })} 
                                min={-180} 
                                max={180} 
                            />
                            <InputSlider 
                                label="Latitude" 
                                value={location.y} 
                                onChange={val => setLocation({ ...location, y: val })} 
                                min={-90} 
                                max={90} 
                            />
                            <InputSlider 
                                label="Height" 
                                value={location.z} 
                                onChange={val => setLocation({ ...location, z: val })} 
                                min={50} 
                                max={200} 
                            />
                        </div>
                    </div>

                    {/* Vertical Card - Linear Velocity */}
                    <div style={{
                        width: '25vw',
                        background: 'rgba(30, 30, 30, 0.3)',
                        backdropFilter: 'blur(20px)',
                        WebkitBackdropFilter: 'blur(20px)',
                        border: '1px solid rgba(255, 255, 255, 0.1)',
                        borderRadius: '8px',
                        padding: '20px',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
                    }}>
                        <h3 style={{
                            margin: '0 0 15px 0',
                            textAlign: 'center',
                            fontWeight: '500',
                            color: '#E0E0E0'
                        }}>Linear Velocity</h3>
                        <div style={{
                            display: 'flex',
                            gap: '40px',
                            flexGrow: 1,
                            width: '100%',
                            justifyContent: 'center',
                            alignItems: 'flex-start',
                            height: '100%',
                            paddingTop: '0px'
                        }}>
                            <InputSlider 
                                label="x" 
                                orientation="vertical" 
                                value={linearVelocity.x} 
                                onChange={val => setLinearVelocity({ ...linearVelocity, x: val })} 
                                min={-0.5} 
                                max={0.5} 
                            />
                            <InputSlider 
                                label="y" 
                                orientation="vertical" 
                                value={linearVelocity.y} 
                                onChange={val => setLinearVelocity({ ...linearVelocity, y: val })} 
                                min={-0.5} 
                                max={0.5} 
                            />
                            <InputSlider 
                                label="z" 
                                orientation="vertical" 
                                value={linearVelocity.z} 
                                onChange={val => setLinearVelocity({ ...linearVelocity, z: val })} 
                                min={-1} 
                                max={-0.5} 
                            />
                        </div>
                    </div>
                </div>
                
                {/* Status Display */}
                <StatusDisplay 
                    status={simulationStatus} 
                    isLoading={statusLoading}
                />
                
                {/* Control Buttons */}
                <div style={{
                    display: 'flex',
                    gap: '20px',
                    alignItems: 'center'
                }}>
                    <button 
                        onClick={handleLaunch}
                        disabled={isLaunching || !backendConnected || !simulationStatus?.model_loaded || simulationStatus?.is_running}
                        style={{
                            padding: '12px 50px',
                            fontSize: '1rem',
                            fontWeight: 'bold',
                            color: '#E0E0E0',
                            background: 'transparent',
                            border: '1px solid rgba(255, 255, 255, 0.3)',
                            borderRadius: '25px',
                            cursor: 'pointer',
                            transition: 'all 0.3s ease',
                            letterSpacing: '1.2px',
                            backdropFilter: 'blur(10px)',
                            WebkitBackdropFilter: 'blur(10px)',
                            opacity: (isLaunching || !backendConnected || !simulationStatus?.model_loaded || simulationStatus?.is_running) ? 0.5 : 1
                        }}
                        onMouseEnter={(e) => {
                            if (!e.target.disabled) {
                                e.target.style.background = 'rgba(255, 255, 255, 0.1)';
                                e.target.style.borderColor = 'rgba(255, 255, 255, 0.5)';
                            }
                        }}
                        onMouseLeave={(e) => {
                            if (!e.target.disabled) {
                                e.target.style.background = 'transparent';
                                e.target.style.borderColor = 'rgba(255, 255, 255, 0.3)';
                            }
                        }}
                    >
                        {isLaunching ? 'LAUNCHING...' : 'LAUNCH'}
                    </button>

                    <button 
                        onClick={handleStop}
                        disabled={!simulationStatus?.is_running}
                        style={{
                            padding: '12px 30px',
                            fontSize: '1rem',
                            fontWeight: 'bold',
                            color: '#E0E0E0',
                            background: 'transparent',
                            border: '1px solid rgba(255, 0, 0, 0.3)',
                            borderRadius: '25px',
                            cursor: 'pointer',
                            transition: 'all 0.3s ease',
                            letterSpacing: '1.2px',
                            backdropFilter: 'blur(10px)',
                            WebkitBackdropFilter: 'blur(10px)',
                            opacity: !simulationStatus?.is_running ? 0.5 : 1
                        }}
                        onMouseEnter={(e) => {
                            if (!e.target.disabled) {
                                e.target.style.background = 'rgba(255, 0, 0, 0.1)';
                                e.target.style.borderColor = 'rgba(255, 0, 0, 0.5)';
                            }
                        }}
                        onMouseLeave={(e) => {
                            if (!e.target.disabled) {
                                e.target.style.background = 'transparent';
                                e.target.style.borderColor = 'rgba(255, 0, 0, 0.3)';
                            }
                        }}
                    >
                        STOP
                    </button>

                    <button 
                        onClick={() => {
                            checkBackendHealth();
                            getSimulationStatus();
                        }}
                        style={{
                            padding: '12px 30px',
                            fontSize: '1rem',
                            fontWeight: 'bold',
                            color: '#E0E0E0',
                            background: 'transparent',
                            border: '1px solid rgba(100, 255, 100, 0.3)',
                            borderRadius: '25px',
                            cursor: 'pointer',
                            transition: 'all 0.3s ease',
                            letterSpacing: '1.2px',
                            backdropFilter: 'blur(10px)',
                            WebkitBackdropFilter: 'blur(10px)'
                        }}
                        onMouseEnter={(e) => {
                            e.target.style.background = 'rgba(100, 255, 100, 0.1)';
                            e.target.style.borderColor = 'rgba(100, 255, 100, 0.5)';
                        }}
                        onMouseLeave={(e) => {
                            e.target.style.background = 'transparent';
                            e.target.style.borderColor = 'rgba(100, 255, 100, 0.3)';
                        }}
                    >
                        REFRESH
                    </button>
                </div>
            </div>
        </div>
    );
}

export default App;