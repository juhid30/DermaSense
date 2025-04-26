import React, { useRef, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import ReactFaceAge from "react-face-age";
import { Camera } from "lucide-react";

function HomePage() {
  const backgroundRef = useRef(null);
  const navigate = useNavigate();
  const [age, setAge] = useState(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Access token for face-age API
  const accessToken = "SJMe2bo0SVRa2q0hy46s";

  // FaceAge options configuration
  const faceAgeOptions = {
    faceageId: accessToken,
    type: 'skincare-analyzer',
    // Other options can be configured here
  };

  useEffect(() => {
    const handleMouseMove = (e) => {
      const { clientX, clientY } = e;
      const { innerWidth, innerHeight } = window;
      const xPercent = (clientX / innerWidth) * 100;
      const yPercent = (clientY / innerHeight) * 100;

      if (backgroundRef.current) {
        backgroundRef.current.style.background = `radial-gradient(circle at ${xPercent}% ${yPercent}%, #e6ccb2, #829bab, #728394)`;
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  const handleCapture = () => {
    setIsCapturing(true);
    setCapturedImage(null);
    setAge(null);
  };

  const handleResult = (result) => {
    if (result && result.image) {
      setCapturedImage(result.image);
      setIsCapturing(false);
      setIsProcessing(true);
      
      // Process the image with face-age API
      setTimeout(() => {
        // Simulating API response for demo purposes
        // In production, you would make an actual API call
        const estimatedAge = Math.floor(Math.random() * 30) + 20; // Random age between 20-50
        setAge(estimatedAge);
        setIsProcessing(false);
      }, 1500);
    }
  };

  const handleRetake = () => {
    setCapturedImage(null);
    setAge(null);
    setIsCapturing(true);
  };

  // Handle FaceAge component load
  const handleFaceAgeLoad = (result) => {
    console.log("FaceAge loaded:", result);
    // You can handle additional setup here if needed
  };

  return (
    <div
      className="min-h-screen bg-gradient-to-b from-[#c0b3a5] to-[#829bab] py-12 px-4 sm:px-6 flex items-center justify-center"
      ref={backgroundRef}
    >
      <div className="max-w-6xl w-full mx-auto bg-[#f5f3f0] rounded-2xl shadow-2xl overflow-hidden flex flex-col md:flex-row">
        {/* Left side with FaceAge component */}
        <div className="w-full md:w-1/2 relative min-h-96 md:h-auto overflow-hidden bg-[#1e1b19]/10 flex flex-col items-center justify-center p-4">
          <div className="relative w-full h-full min-h-80 rounded-lg overflow-hidden flex flex-col items-center justify-center">
            {isCapturing ? (
              <div className="w-full h-full min-h-80 flex flex-col items-center justify-center">
                <div className="w-full h-full min-h-80">
                  <ReactFaceAge
                    options={faceAgeOptions}
                    showCamera={true}
                    showFacePoint={true}
                    onResult={handleResult}
                    onLoad={handleFaceAgeLoad}
                    className="w-full h-full"
                    style={{ width: '100%', height: '100%', minHeight: '320px' }}
                  />
                </div>
              </div>
            ) : capturedImage ? (
              <div className="flex flex-col items-center justify-center w-full h-full">
                <div className="relative w-full h-64 md:h-80 rounded-lg overflow-hidden">
                  <img 
                    src={capturedImage} 
                    alt="Captured face" 
                    className="w-full h-full object-cover object-center"
                  />
                </div>
                
                <div className="mt-4 flex flex-col items-center">
                  {isProcessing ? (
                    <div className="p-4 bg-[#e6ccb2]/30 rounded-lg flex items-center justify-center">
                      <svg className="animate-spin h-5 w-5 text-[#6d4f3e] mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <p className="text-[#6d4f3e] font-light">Analyzing skin age...</p>
                    </div>
                  ) : (
                    <div className="p-4 bg-[#e6ccb2]/30 rounded-lg text-center">
                      <p className="text-[#6d4f3e] font-light">Estimated skin age: <span className="font-medium">{age}</span></p>
                      <p className="text-xs text-[#6d4f3e]/80 mt-1">Based on facial analysis</p>
                    </div>
                  )}
                  
                  <div className="mt-4 flex space-x-3">
                    <button
                      onClick={handleRetake}
                      className="bg-[#829bab] hover:bg-[#728394] text-white font-light py-2 px-4 rounded-lg shadow-md transition duration-300 transform hover:scale-105 uppercase tracking-wider text-sm flex items-center"
                    >
                      <Camera className="w-4 h-4 mr-2" />
                      Retake
                    </button>
                    
                    <button
                      onClick={() => navigate("/quiz")}
                      className="bg-[#6d4f3e] hover:bg-[#5d3f2e] text-white font-light py-2 px-4 rounded-lg shadow-md transition duration-300 transform hover:scale-105 uppercase tracking-wider text-sm"
                    >
                      Get Recommendations
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center space-y-4 text-center p-6">
                <div className="w-24 h-24 rounded-full bg-[#829bab]/30 flex items-center justify-center">
                  <Camera className="w-12 h-12 text-[#728394]" />
                </div>
                <h3 className="text-xl font-light text-[#1e1b19] tracking-wide">
                  Try Our Age Detection
                </h3>
                <p className="text-sm text-[#728394] max-w-xs">
                  Experience DermaSense's AI technology with a quick preview of our skin age detection feature
                </p>
                <button
                  onClick={handleCapture}
                  className="mt-4 bg-[#829bab] hover:bg-[#728394] text-white font-light py-3 px-6 rounded-lg shadow-md transition duration-300 transform hover:scale-105 uppercase tracking-wider text-sm flex items-center"
                >
                  <Camera className="w-4 h-4 mr-2" />
                  Start Camera
                </button>
              </div>
            )}
            <div className="absolute inset-0 bg-gradient-to-r from-[#1e1b19]/20 to-transparent pointer-events-none"></div>
          </div>
        </div>

        {/* Right side with content */}
        <div className="w-full md:w-1/2 p-8 md:p-12 flex flex-col justify-center">
          <h1 className="text-4xl md:text-5xl font-light text-[#1e1b19] mb-6 tracking-wider">
            DermaSense
          </h1>

          <p className="text-lg md:text-xl text-[#728394] mb-10 font-light">
            Your personalized skin care assistant powered by AI
          </p>

          <button
            onClick={() => navigate("/quiz")}
            className="w-full bg-[#829bab] hover:bg-[#728394] text-white font-light py-5 px-8 rounded-lg shadow-lg transition duration-300 transform hover:scale-105 uppercase tracking-wider text-lg"
          >
            Take Quiz
          </button>

          <div className="mt-8 border-t border-[#c0b3a5] pt-6">
            <p className="text-sm text-[#6d4f3e] font-light">
              Discover your personalized skincare routine in just a few minutes
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HomePage;
