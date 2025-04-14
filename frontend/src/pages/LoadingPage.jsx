import React from "react";

function LoadingPage() {
  return (
    <div className="flex flex-col items-center justify-center h-screen w-screen bg-almond-500">
      {/* Spinner */}
      <div className="w-16 h-16 border-4 border-t-tan-400 border-b-tan-400 border-l-transparent border-r-transparent rounded-full animate-spin mb-6"></div>

      {/* Text */}
      <div className="text-2xl text-coffee-600 font-medium animate-pulse">
        Please wait...
      </div>
    </div>
  );
}

export default LoadingPage;
