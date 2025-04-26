import React from "react";

function LoadingPage() {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 via-amber-50 to-blue-100">
      {/* Container with beige border */}
      <div className="bg-white bg-opacity-80 rounded-2xl shadow-xl p-12 flex flex-col items-center border-2 border-[#ddb892]">
        {/* Spinner */}
        <div className="relative w-24 h-24 mb-8">
          <div className="absolute inset-0 border-4 border-[#e6ccb2] border-t-[#6d4f3e] rounded-full animate-spin"></div>
          <div className="absolute inset-2 border-4 border-[#829bab] border-b-[#728394] rounded-full animate-spin-slow"></div>
        </div>

        {/* Text */}
        <div className="text-center">
          <p className="text-xl font-serif text-amber-900 tracking-wider">Please wait...</p>
          <p className="text-sm text-blue-800 mt-2 font-light">Analyzing your skin profile</p>
        </div>
      </div>
      
      {/* Style for custom animation speed */}
      <style jsx="true">{`
        @keyframes spin-slow {
          to {
            transform: rotate(-360deg);
          }
        }
        .animate-spin-slow {
          animation: spin-slow 3s linear infinite;
        }
      `}</style>
    </div>
  );
}

export default LoadingPage