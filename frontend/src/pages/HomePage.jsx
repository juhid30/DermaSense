import React, { useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";

function HomePage() {
  const backgroundRef = useRef(null);
  const navigate = useNavigate();

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

  return (
    <div 
      className="min-h-screen bg-gradient-to-b from-[#c0b3a5] to-[#829bab] py-12 px-4 sm:px-6 flex items-center justify-center"
      ref={backgroundRef}
    >
      <div className="max-w-6xl w-full mx-auto bg-[#f5f3f0] rounded-2xl shadow-2xl overflow-hidden flex flex-col md:flex-row">
        {/* Left side with background image */}
        <div className="w-full md:w-1/2 relative h-72 md:h-auto overflow-hidden">
          <img 
            src="/face.jpg" 
            alt="Woman applying skincare" 
            className="absolute inset-0 w-full h-full object-cover object-center"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-[#1e1b19]/30 to-transparent"></div>
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