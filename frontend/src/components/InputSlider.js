import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Slider from '@mui/material/Slider';
import MuiInput from '@mui/material/Input';

export default function InputSlider({ label, value, onChange, orientation = 'horizontal', min = 0, max = 100 }) {

    const handleSliderChange = (event, newValue) => {
        onChange(newValue);
    };

    const handleInputChange = (event) => {
        onChange(event.target.value === '' ? '' : Number(event.target.value));
    };

    const handleBlur = () => {
        if (value < min) {
            onChange(min);
        } else if (value > max) {
            onChange(max);
        }
    };

    // Styles for blue sliders on a dark theme
    const sliderSx = {
        color: '#2979FF', // accent-blue
        '& .MuiSlider-rail': {
            opacity: 0.5,
            backgroundColor: '#424242', // border-light
        },
    };

    // Styles for a visible input box on a dark theme - made wider to accommodate larger numbers
    const inputSx = {
        width: 100, // Increased to 100px to fully accommodate 5-digit numbers
        height: 32,
        backgroundColor: '#121212', // background-dark
        color: '#E0E0E0', // text-light
        border: '1px solid #424242', // border-light
        borderRadius: '4px',
        '& input': {
            textAlign: 'center',
            padding: '4px 8px', // Added horizontal padding for better spacing
            fontSize: '0.875rem', // Slightly smaller font to fit more digits
        }
    };

    if (orientation === 'vertical') {
        return (
            <Box sx={{ height: '100%', display: 'flex', gap: 0.05, flexDirection: 'column', alignItems: 'center' }}>
                <Slider
                    orientation="vertical"
                    value={typeof value === 'number' ? value : min}
                    onChange={handleSliderChange}
                    min={min}
                    max={max}
                    sx={{ ...sliderSx, flexGrow: 1 }}
                />
                <MuiInput
                    value={value}
                    size="small"
                    onChange={handleInputChange}
                    onBlur={handleBlur}
                    disableUnderline
                    inputProps={{ step: 1, min, max, type: 'number' }}
                    sx={{...inputSx, my: 2}}
                />
                <Typography gutterBottom sx={{color: '#A0A0A0'}}>{label}</Typography>
            </Box>
        );
    }

    // Horizontal Slider Layout
    return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, width: '100%' }}>
            {label && (
                <Typography sx={{ width: '80px', flexShrink: 0, color: '#A0A0A0' }}>
                    {label}
                </Typography>
            )}
            <Slider
                value={typeof value === 'number' ? value : min}
                onChange={handleSliderChange}
                min={min}
                max={max}
                sx={sliderSx}
            />
            <MuiInput
                value={value}
                size="small"
                onChange={handleInputChange}
                onBlur={handleBlur}
                disableUnderline
                inputProps={{ step: 1, min, max, type: 'number' }}
                sx={inputSx}
            />
        </Box>
    );
}