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
        "http://127.0.0.1:5000/predict-face",
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

    backgroundRef.current.style.background = `radial-gradient(circle at ${xPercent}% ${yPercent}%, #e1d5c4, #c0b3a5, #728394)`;
  };

  return (
    <div
      ref={backgroundRef}
      onMouseMove={handleMouseMove}
      className="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 transition-all duration-500 ease-in-out"
      style={{
        background:
          "radial-gradient(circle at center, #e1d5c4, #c0b3a5, #728394)",
        fontFamily: "'Playfair Display', serif",
      }}
    >
      <div className="max-w-6xl mx-auto bg-white bg-opacity-95 backdrop-blur-lg rounded-3xl shadow-2xl overflow-hidden relative z-10 border border-[#c0b3a5]">
        {/* Header with gradient background and light text */}
        <div className="bg-gradient-to-r from-[#1e1b19] to-[#728394] py-8 px-10 flex justify-between items-center rounded-t-2xl border-b-4 border-[#c0b3a5]">
          <h2 className="text-4xl font-serif font-bold text-[#e1d5c4] tracking-wider">
            Skin Analysis
          </h2>
          <button
            onClick={() => navigate("/dashboard")}
            className="text-[#e1d5c4] hover:text-white transition-colors transform hover:scale-105 flex items-center bg-[#6d4f3e] bg-opacity-40 px-4 py-2 rounded-full"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 mr-2"
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

        <div className="grid grid-cols-1 md:grid-cols-2 gap-10 p-10">
          <div className="flex flex-col space-y-8">
            {/* Camera capture area with rounded card */}
            <div className="bg-gradient-to-b from-[#e1d5c4] to-[#c0b3a5] rounded-2xl overflow-hidden h-72 md:h-96 flex items-center justify-center relative transition-all duration-500 hover:shadow-2xl border border-[#c0b3a5]">
              {cameraActive ? (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover transition-all duration-300 ease-in-out"
                ></video>
              ) : (
                <div className="text-[#728394] flex flex-col items-center justify-center opacity-75">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-20 w-20 mb-4 animate-pulse"
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
                  <p className="text-2xl font-serif">Camera not active</p>
                </div>
              )}
            </div>

            <div className="flex gap-4">
              {/* Button Styles with hover scale */}
              <button
                onClick={captureImage}
                className="bg-gradient-to-r from-[#6d4f3e] to-[#728394] text-white font-medium py-4 px-8 rounded-full transition duration-300 transform hover:scale-105 hover:shadow-lg flex-grow flex items-center justify-center"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 mr-2"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M4 5a2 2 0 00-2 2v8a2 2 0 002 2h12a2 2 0 002-2V7a2 2 0 00-2-2h-1.586a1 1 0 01-.707-.293l-1.121-1.121A2 2 0 0011.172 3H8.828a2 2 0 00-1.414.586L6.293 4.707A1 1 0 015.586 5H4zm6 9a3 3 0 100-6 3 3 0 000 6z"
                    clipRule="evenodd"
                  />
                </svg>
                Capture Image
              </button>
              <button
                onClick={triggerFileInput}
                className="bg-gradient-to-r from-[#829bab] to-[#c0b3a5] text-white font-medium py-4 px-8 rounded-full transition duration-300 transform hover:scale-105 hover:shadow-lg flex-grow flex items-center justify-center"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 mr-2"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
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

          <div className="bg-gradient-to-b from-white to-[#e1d5c4] rounded-2xl p-8 flex flex-col items-center transform transition-all duration-300 hover:shadow-xl border border-[#c0b3a5]">
            <h3 className="text-2xl font-serif font-semibold text-[#1e1b19] mb-6 tracking-wide flex items-center">
              <span className="mr-3 bg-[#728394] text-white p-2 rounded-full">
                📷
              </span>
              Your Image
            </h3>
            {images.length === 0 ? (
              <div className="h-64 w-full flex items-center justify-center text-[#728394] bg-white bg-opacity-70 rounded-xl border border-dashed border-[#c0b3a5] p-6">
                <div className="text-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-16 w-16 mx-auto mb-4 opacity-50"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                  <p className="text-lg font-serif">
                    No image yet. Capture or upload to begin your skin analysis.
                  </p>
                </div>
              </div>
            ) : (
              <div className="relative w-full mb-8 group">
                {/* Image with rounded corners */}
                <img
                  src={images[0]}
                  alt="Uploaded"
                  className="rounded-xl shadow-lg w-full object-contain max-h-64 transition-all duration-500 ease-in-out border-2 border-[#c0b3a5]"
                />
                {/* X button appears on hover with transition */}
                <button
                  onClick={removeImage}
                  className="absolute top-2 right-2 bg-[#1e1b19] text-white rounded-full p-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 transform hover:scale-110 shadow-md"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                      clipRule="evenodd"
                    />
                  </svg>
                </button>
              </div>
            )}

            {images.length > 0 && (
              <button
                onClick={analyzeImages}
                disabled={analyzing}
                className={`${
                  analyzing
                    ? "bg-[#c0b3a5] cursor-not-allowed"
                    : "bg-gradient-to-r from-[#6d4f3e] to-[#728394] hover:from-[#1e1b19] hover:to-[#728394] text-white"
                } font-medium py-4 px-8 rounded-full transition duration-300 w-full transform hover:scale-105 shadow-lg flex items-center justify-center text-lg font-serif`}
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
                  <>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5 mr-2"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M9 3a1 1 0 012 0v5.5a.5.5 0 001 0V4a1 1 0 112 0v4.5a.5.5 0 001 0V6a1 1 0 112 0v5a7 7 0 11-14 0V9a1 1 0 012 0v2.5a.5.5 0 001 0V4a1 1 0 012 0v4.5a.5.5 0 001 0V3z"
                        clipRule="evenodd"
                      />
                    </svg>
                    Analyze Skin
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        <div className="bg-[#e1d5c4] bg-opacity-50 py-6 px-10 text-center border-t border-[#c0b3a5]">
          <p className="text-[#6d4f3e] font-serif">
            For best results, ensure good lighting and a clear view of the skin
            area you wish to analyze.
          </p>
        </div>
      </div>

      <div className="mt-8 text-center text-[#1e1b19] opacity-70 font-serif absolute bottom-4 left-0 right-0">
        © 2025 Skin Analysis System • All Rights Reserved
      </div>
    </div>
  );
}

export default CameraPage;
