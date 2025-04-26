import React, { useRef, useState, useEffect } from 'react';
import * as faceapi from 'face-api.js';

const LiveFaceAgeDetector = () => {
  const videoRef = useRef();
  const canvasRef = useRef();
  const [age, setAge] = useState(null);
  const [gender, setGender] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [error, setError] = useState(null);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);

  // Load models on component mount
  useEffect(() => {
    const loadModels = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        // Change model loading to specify exact model paths and add more debugging
        console.log('Starting model loading...');
        
        // For SSD MobileNet
        console.log('Loading SSD MobileNet model...');
        await faceapi.nets.ssdMobilenetv1.loadFromUri('/models/ssd_mobilenetv1');
        console.log('SSD MobileNet loaded successfully');
        
        // For Face Landmark
        console.log('Loading Landmark model...');
        await faceapi.nets.faceLandmark68Net.loadFromUri('/models/face_landmark_68');
        console.log('Landmark model loaded successfully');
        
        // For Age and Gender
        console.log('Loading Age/Gender model...');
        await faceapi.nets.ageGenderNet.loadFromUri('/models/age_gender_model');
        console.log('Age/Gender model loaded successfully');
        
        setModelsLoaded(true);
        console.log('All models loaded successfully');
      } catch (err) {
        // More detailed error handling
        console.error('Error loading models:', err);
        console.error('Error name:', err.name);
        console.error('Error message:', err.message);
        
        let errorMessage = `Failed to load models: ${err.message}`;
        
        // Check if we're getting HTML instead of JSON (common 404 symptom)
        if (err.message.includes("Unexpected token '<'")) {
          errorMessage = 'Model files not found. Please ensure models are in the correct location.';
        } else if (err.message.includes("tensor should have")) {
          errorMessage = 'Model files appear to be corrupted or incompatible. Try downloading fresh model files.';
        }
        
        setError(errorMessage);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadModels();
    
    // Cleanup on unmount
    return () => {
      stopCamera();
    };
  }, []);

  // Start webcam
  const startCamera = async () => {
    if (!modelsLoaded) {
      setError('Models are still loading. Please wait.');
      return;
    }
    
    setError(null);
    setIsLoading(true);
    
    try {
      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' }
      });
      
      // Store stream reference for cleanup
      streamRef.current = stream;
      
      // Set video source to stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        
        // Wait for video to be ready
        await new Promise((resolve) => {
          videoRef.current.onloadedmetadata = resolve;
          videoRef.current.onloadeddata = resolve; // Backup in case onloadedmetadata doesn't fire
        });
        
        // Ensure video is playing
        await videoRef.current.play();
        
        // Start processing
        setIsStreamActive(true);
        startFaceDetection();
      }
    } catch (err) {
      let errorMessage = `Camera error: ${err.message}`;
      
      // More user-friendly messages for common errors
      if (err.name === 'NotAllowedError') {
        errorMessage = 'Camera access denied. Please allow camera access and try again.';
      } else if (err.name === 'NotFoundError') {
        errorMessage = 'No camera found. Please connect a camera and try again.';
      }
      
      setError(errorMessage);
      console.error('Camera error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Stop webcam
  const stopCamera = () => {
    // Stop face detection interval
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
      detectionIntervalRef.current = null;
    }
    
    // Stop media tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    // Clear video source
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject = null;
    }
    
    // Clear canvas
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    
    setIsStreamActive(false);
    setAge(null);
    setGender(null);
  };

  // Toggle camera
  const toggleCamera = () => {
    if (isStreamActive) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  // Start continuous face detection
  const startFaceDetection = () => {
    // Run detection every 200ms
    detectionIntervalRef.current = setInterval(async () => {
      if (videoRef.current && canvasRef.current && modelsLoaded && 
          videoRef.current.readyState === 4 && !videoRef.current.paused) {
        await detectFaces();
      }
    }, 200);
  };

  // Detect faces in video frame
  const detectFaces = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    try {
      // Use SSD MobileNet for detection
      const detections = await faceapi
        .detectAllFaces(videoRef.current)
        .withFaceLandmarks()
        .withAgeAndGender();
      
      if (detections.length === 0) {
        setAge(null);
        setGender(null);
        return;
      }
      
      // Draw results on canvas
      const displaySize = {
        width: videoRef.current.videoWidth,
        height: videoRef.current.videoHeight
      };
      
      // Make sure canvas dimensions match video
      faceapi.matchDimensions(canvasRef.current, displaySize);
      
      // Resize detections to match display size
      const resizedDetections = faceapi.resizeResults(detections, displaySize);
      
      // Clear previous drawings
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // Draw face detections with age and gender
      resizedDetections.forEach(detection => {
        const { age, gender, genderProbability } = detection;
        const box = detection.detection.box;
        
        // Draw detection box
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: `${Math.round(age)} years, ${gender} (${Math.round(genderProbability * 100)}%)`
        });
        drawBox.draw(canvasRef.current);
      });
      
      // Use first face for main display if available
      if (detections.length > 0) {
        setAge(Math.round(detections[0].age));
        setGender(detections[0].gender);
      }
    } catch (err) {
      console.error('Error in face detection:', err);
      setError(`Detection error: ${err.message}`);
    }
  };

  return (
    <div className="flex flex-col items-center text-white p-6 bg-black min-h-screen">
      <h1 className="text-2xl font-bold mb-6 text-blue-400">Live Face Age Detector</h1>
      
      {error && (
        <div className="bg-red-900 text-white p-3 rounded-lg mb-4 w-full max-w-md">
          {error}
        </div>
      )}
      
      <div className="flex flex-col md:flex-row gap-6 mb-6 w-full max-w-3xl">
        <div className="flex-1">
          <div className="bg-gray-900 p-4 rounded-lg mb-4">
            <h2 className="text-xl mb-4 text-blue-300">Control</h2>
            
            <button 
              onClick={toggleCamera}
              disabled={!modelsLoaded || isLoading}
              className={`w-full p-4 rounded-lg font-bold ${
                isStreamActive 
                  ? "bg-red-600 hover:bg-red-700" 
                  : "bg-green-600 hover:bg-green-700"
              } disabled:bg-gray-700 disabled:cursor-not-allowed`}
            >
              {isStreamActive ? "Stop Camera" : "Start Camera"}
            </button>
            
            {!modelsLoaded && (
              <div className="mt-4 text-yellow-400">
                Loading AI models... Please wait.
              </div>
            )}
            
            {isLoading && (
              <div className="mt-4 text-blue-400">
                Processing... Please wait.
              </div>
            )}
            
            <div className="mt-4 text-gray-400 text-sm">
              <p>Model Status: {modelsLoaded ? "Loaded ✅" : "Loading..."}</p>
              <p>Camera Status: {isStreamActive ? "Active ✅" : "Inactive"}</p>
            </div>
          </div>
          
          {age !== null && (
            <div className="bg-gray-900 p-4 rounded-lg">
              <h2 className="text-xl mb-4 text-blue-300">Results</h2>
              <div className="space-y-2">
                <p className="text-lg">Estimated Age: <span className="font-bold text-green-400">{age} years</span></p>
                {gender && (
                  <p className="text-lg">Gender: <span className="font-bold text-green-400">{gender}</span></p>
                )}
              </div>
            </div>
          )}
          
          <div className="bg-gray-900 p-4 rounded-lg mt-4">
            <h2 className="text-xl mb-4 text-blue-300">Troubleshooting</h2>
            <ul className="list-disc pl-5 text-gray-300 text-sm space-y-2">
              <li>Make sure your face-api.js model files are in the correct location: <code className="bg-gray-800 px-1 rounded">/public/models/</code></li>
              <li>Each model folder should contain proper model files (usually .bin and .json files)</li>
              <li>Ensure good lighting for better detection</li>
              <li>Position your face clearly in the frame</li>
              <li>Allow camera permissions when prompted</li>
              <li>Try refreshing the page if models don't load correctly</li>
            </ul>
          </div>
        </div>
        
        <div className="flex-1 bg-gray-900 p-4 rounded-lg">
          <h2 className="text-xl mb-4 text-blue-300">Live Camera</h2>
          <div className="relative rounded-lg overflow-hidden bg-gray-800">
            {isLoading && (
              <div className="absolute inset-0 bg-black bg-opacity-70 flex items-center justify-center z-20">
                <div className="text-white">Processing...</div>
              </div>
            )}
            
            {/* Video element */}
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full h-auto"
              style={{ display: isStreamActive ? 'block' : 'none' }}
            />
            
            {/* Canvas overlay for face detection drawing */}
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full"
              style={{ display: isStreamActive ? 'block' : 'none' }}
            />
            
            {!isStreamActive && (
              <div className="w-full h-64 flex items-center justify-center">
                <p className="text-gray-400">Camera is off</p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className="text-gray-400 text-sm mt-4 max-w-2xl">
        <p>
          This application uses face-api.js to detect faces and estimate age and gender from your webcam feed in real-time.
          The estimation works best in good lighting conditions with a clear view of your face.
        </p>
        <p className="mt-2">
          All processing happens locally in your browser - no images are sent to any server.
        </p>
      </div>
    </div>
  );
};

export default LiveFaceAgeDetector;