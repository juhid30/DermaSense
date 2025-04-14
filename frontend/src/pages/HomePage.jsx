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
        backgroundRef.current.style.background = `radial-gradient(circle at ${xPercent}% ${yPercent}%, #e6ccb2, #ddb892, #b08968)`;
      }
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  return (
    <div
      ref={backgroundRef}
      className="flex flex-col items-center justify-center h-screen w-screen transition-all duration-300"
      style={{
        background:
          "radial-gradient(circle at center, #e6ccb2, #ddb892, #b08968)",
      }}
    >
      <h1 className="text-5xl md:text-7xl font-serif italic font-semibold text-stone-800 drop-shadow-md mb-8 tracking-tight">
        Derma<span className="italic font-light">Sense</span>
      </h1>
      <p className="text-xl text-stone-600 mb-12 max-w-2xl text-center px-4">
        Your personalized skin care assistant powered by AI
      </p>
      <button
        onClick={() => navigate("/quiz")}
        className="border-2 border-amber-700 text-amber-700 hover:bg-amber-700 hover:text-white font-medium py-3 px-16 rounded-full text-lg transition-all duration-300 transform hover:scale-105 w-64"
      >
        Take Quiz
      </button>
    </div>
  );
}

export default HomePage;
