import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
function CameraPage() {
  const [images, setImages] = useState([]);
  const [cameraActive, setCameraActive] = useState(false);
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const backgroundRef = useRef(null);
  const navigate = useNavigate();

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      setStream(mediaStream);
      setCameraActive(true);
    } catch (error) {
      console.error("Error accessing camera:", error);
      alert("Unable to access camera. Please check your permissions.");
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    setCameraActive(false);
  };

  const captureImage = () => {
    if (videoRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      const imageURL = canvas.toDataURL("image/jpeg");
      setImages([imageURL]);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      setImages([event.target.result]);
    };
    reader.readAsDataURL(file);
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const analyzeImages = async () => {
    if (images.length === 0) {
      alert("Please capture or upload at least one image first.");
      return;
    }

    setAnalyzing(true);

    try {
      const base64Response = await fetch(images[0]);
      const blob = await base64Response.blob();

      const formData = new FormData();
      formData.append("image", blob, "captured.jpg");
      const skinAnalysisResults = localStorage.getItem("skinAnalysisResults");

      // Optionally, you can send the skinAnalysisResults along with the form data as an additional field
      formData.append("skinAnalysisResults", skinAnalysisResults);

      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      console.log("Response from API:", response.data); // Still console logging
      navigate("/results", { state: { results: response.data } }); // Redirect with data
    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("Failed to analyze image. Please try again.");
    } finally {
      setAnalyzing(false);
    }
  };

  const removeImage = () => {
    setImages([]);
  };

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  const handleMouseMove = (e) => {
    if (!backgroundRef.current) return;
    const { clientX, clientY } = e;
    const { innerWidth, innerHeight } = window;
    const xPercent = (clientX / innerWidth) * 100;
    const yPercent = (clientY / innerHeight) * 100;

    backgroundRef.current.style.background = `radial-gradient(circle at ${xPercent}% ${yPercent}%, #e6ccb2, #ddb892, #b08968)`;
  };

  return (
    <div
      ref={backgroundRef}
      onMouseMove={handleMouseMove}
      className="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 transition-all duration-500 ease-in-out"
      style={{
        background:
          "radial-gradient(circle at center, #e6ccb2, #ddb892, #b08968)",
      }}
    >
      <div className="max-w-6xl mx-auto bg-white bg-opacity-90 backdrop-blur-sm rounded-3xl shadow-2xl overflow-hidden relative z-10 border border-tan-400">
        {/* Header with gradient background and light text */}
        <div className="bg-gradient-to-r from-coffee-700 to-raw_umber-600 py-6 px-8 flex justify-between items-center rounded-t-2xl">
          <h2 className="text-2xl font-serif font-bold text-white tracking-wider">
            Skin Analysis
          </h2>
          <button
            onClick={() => navigate("/dashboard")}
            className="text-white hover:text-tan-200 transition-colors transform hover:scale-105 flex items-center"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 mr-1"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M9.707 14.707a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 1.414L7.414 9H15a1 1 0 110 2H7.414l2.293 2.293a1 1 0 010 1.414z"
                clipRule="evenodd"
              />
            </svg>
            Dashboard
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 p-8">
          <div className="flex flex-col space-y-6">
            {/* Camera capture area with rounded card */}
            <div className="bg-gradient-to-b from-raw_umber-100 to-tan-200 rounded-2xl overflow-hidden h-64 md:h-80 flex items-center justify-center relative transition-all duration-500 hover:shadow-2xl border border-tan-300">
              {cameraActive ? (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover transition-all duration-300 ease-in-out"
                ></video>
              ) : (
                <div className="text-coffee-500 flex flex-col items-center justify-center opacity-75">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-16 w-16 mb-2 animate-pulse"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
                    />
                  </svg>
                  <p className="text-xl">Camera not active</p>
                </div>
              )}
            </div>

            <div className="flex gap-3">
              {/* Button Styles with hover scale */}
              <button
                onClick={captureImage}
                className="bg-gradient-to-r from-raw_umber-500 to-coffee-500 text-white font-medium py-3 px-8 rounded-full transition duration-300 transform hover:scale-105 hover:shadow-lg flex-grow"
              >
                Capture Image
              </button>
              <button
                onClick={triggerFileInput}
                className="bg-gradient-to-r from-tan-400 to-dun-400 text-white font-medium py-3 px-8 rounded-full transition duration-300 transform hover:scale-105 hover:shadow-lg flex-grow"
              >
                Upload Image
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileUpload}
                accept="image/*"
                className="hidden"
              />
            </div>
          </div>

          <div className="bg-gradient-to-b from-almond-100 to-almond-300 bg-opacity-50 rounded-2xl p-6 flex flex-col items-center transform transition-all duration-300 hover:shadow-xl border border-almond-400">
            <h3 className="text-xl font-serif font-semibold text-coffee-700 mb-4 tracking-wide">
              Your Image
            </h3>
            {images.length === 0 ? (
              <div className="h-56 w-full flex items-center justify-center text-coffee-500 bg-white bg-opacity-70 rounded-xl border border-dashed border-coffee-300">
                <p className="text-center px-4">
                  No image yet. Capture or upload to begin your skin analysis.
                </p>
              </div>
            ) : (
              <div className="relative w-full mb-6 group">
                {/* Image with rounded corners */}
                <img
                  src={images[0]}
                  alt="Uploaded"
                  className="rounded-xl shadow-md w-full object-contain max-h-64 transition-all duration-500 ease-in-out border border-coffee-200"
                />
                {/* X button appears on hover with transition */}
                <button
                  onClick={removeImage}
                  className="absolute top-2 right-2 bg-red-500 text-white rounded-full p-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 transform hover:scale-110 shadow-md"
                >
                  ✕
                </button>
              </div>
            )}

            {images.length > 0 && (
              <button
                onClick={analyzeImages}
                disabled={analyzing}
                className={`${
                  analyzing
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-gradient-to-r from-coffee-600 to-raw_umber-500 hover:from-coffee-700 hover:to-raw_umber-600 text-white"
                } font-medium py-3 px-8 rounded-full transition duration-300 w-full transform hover:scale-105 shadow-md`}
              >
                {analyzing ? (
                  <>
                    <svg
                      className="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline-block"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Analyzing...
                  </>
                ) : (
                  "Analyze Image"
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default CameraPage;
