import React, { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";

function ResultsPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [saved, setSaved] = useState(false);
  const backgroundRef = useRef(null);

  // Get current date and time for timestamping
  const currentDate = new Date().toLocaleString();
  console.log(location.state.results);
  // Get any passed data from navigation (for future backend integration)
  const resultsData = location.state?.results || null;
  if (!resultsData) {
    return <div>No results available.</div>;
  }

  // Check if previous results exist in localStorage when component mounts
  useEffect(() => {
    console.log(resultsData);
    const savedResults = localStorage.getItem("skinAnalysisResults");
    if (savedResults) {
      console.log("Previously saved results found:", JSON.parse(savedResults));
    }
  }, []);

  const saveResults = () => {
    try {
      // Fetch existing results from localStorage
      const existingDataString = localStorage.getItem("skinAnalysisResults");
      let allResults = [];

      if (existingDataString) {
        // Parse existing data if it exists
        allResults = JSON.parse(existingDataString);
        // Ensure it's an array
        if (!Array.isArray(allResults)) {
          allResults = [allResults]; // Convert to array if it was a single object
        }
      }

      // Add current results to the array
      allResults.push(resultsData);

      // Save the updated array back to localStorage
      localStorage.setItem("skinAnalysisResults", JSON.stringify(allResults));

      console.log("Results appended to localStorage:", resultsData);
      console.log("All saved results:", allResults);

      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (error) {
      console.error("Error saving to localStorage:", error);
      alert("Unable to save results. Please try again.");
    }
  };

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
      className="min-h-screen py-12 px-4 sm:px-6 transition-all duration-500 ease-in-out"
      style={{
        background:
          "radial-gradient(circle at center, #e1d5c4, #c0b3a5, #728394)",
        fontFamily: "'Playfair Display', serif",
      }}
    >
      <div className="max-w-5xl mx-auto">
        <div className="bg-white bg-opacity-95 backdrop-blur-lg rounded-3xl shadow-2xl overflow-hidden border border-[#c0b3a5]">
          <div className="bg-gradient-to-r from-[#1e1b19] to-[#728394] py-8 px-10 flex justify-between items-center rounded-t-2xl border-b-4 border-[#c0b3a5]">
            <h2 className="text-4xl font-serif font-bold text-[#e1d5c4] tracking-wider">
              Skin Analysis Results
            </h2>
            <div className="flex space-x-6">
              <button
                onClick={() => navigate("/camera")}
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
                    d="M4 5a2 2 0 00-2 2v8a2 2 0 002 2h12a2 2 0 002-2V7a2 2 0 00-2-2h-1.586a1 1 0 01-.707-.293l-1.121-1.121A2 2 0 0011.172 3H8.828a2 2 0 00-1.414.586L6.293 4.707A1 1 0 015.586 5H4zm6 9a3 3 0 100-6 3 3 0 000 6z"
                    clipRule="evenodd"
                  />
                </svg>
                New Analysis
              </button>
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
                  <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z" />
                </svg>
                Dashboard
              </button>
            </div>
          </div>

          <div className="p-10 space-y-10">
            {/* Analysis Date/Time */}
            <div className="text-sm text-[#728394] font-medium border-b border-[#c0b3a5] pb-2">
              Analysis performed: {currentDate}
            </div>

            {/* Prediction Results */}
            <section>
              <h3 className="text-2xl font-serif font-semibold text-[#1e1b19] mb-5 flex items-center">
                <span className="mr-3 bg-[#728394] text-white p-2 rounded-full">
                  🧪
                </span>
                Prediction Results
              </h3>
              <div className="p-6 rounded-2xl border border-[#c0b3a5] shadow-lg hover:shadow-xl transition-all duration-300 bg-gradient-to-br from-[#e1d5c4] to-white">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                  <div className="p-5 bg-white bg-opacity-70 rounded-xl shadow-md border-l-4 border-[#728394]">
                    <p className="font-medium text-[#6d4f3e] mb-3 uppercase tracking-wider text-sm">
                      Detected Condition:
                    </p>
                    <p className="text-[#1e1b19] text-xl font-serif">
                      Acne: {resultsData.predicted_label_disease.severity}
                    </p>
                  </div>

                  <div className="p-5 bg-white bg-opacity-70 rounded-xl shadow-md border-l-4 border-[#728394]">
                    <p className="font-medium text-[#6d4f3e] mb-3 uppercase tracking-wider text-sm">
                      Identified Concerns:
                    </p>
                    <p className="text-[#1e1b19] text-xl font-serif">
                      {resultsData.predicted_label_ageing}
                    </p>
                  </div>
                </div>
              </div>
            </section>

            {/* AI Response */}
            <section>
              <h3 className="text-2xl font-serif font-semibold text-[#1e1b19] mb-5 flex items-center">
                <span className="mr-3 bg-[#728394] text-white p-2 rounded-full">
                  🤖
                </span>
                Analysis Summary
              </h3>
              <div className="p-6 rounded-2xl border border-[#c0b3a5] shadow-lg hover:shadow-xl transition-all duration-300 bg-gradient-to-br from-white to-[#e1d5c4]">
                <p className="text-[#1e1b19] leading-relaxed text-lg font-serif">
                  {resultsData.gemini_response.summary ||
                    "No analysis summary available."}
                </p>
              </div>
            </section>

            {/* Recommended Ingredients */}
            <section>
              <h3 className="text-2xl font-serif font-semibold text-[#1e1b19] mb-5 flex items-center">
                <span className="mr-3 bg-[#728394] text-white p-2 rounded-full">
                  🌿
                </span>
                Recommended Active Ingredients
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {resultsData.gemini_response.recommendations.coreIngredients.map(
                  (ingredient, idx) => (
                    <div
                      key={idx}
                      className="p-5 rounded-xl border border-[#c0b3a5] shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-105 bg-white"
                    >
                      <h4 className="font-medium text-[#1e1b19] font-serif">
                        {ingredient}
                      </h4>
                    </div>
                  )
                )}
              </div>
            </section>

            {/* Recommended Foods */}
            <section>
              <h3 className="text-2xl font-serif font-semibold text-[#1e1b19] mb-5 flex items-center">
                <span className="mr-3 bg-[#728394] text-white p-2 rounded-full">
                  🍎
                </span>
                Recommended Foods
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {resultsData.gemini_response.recommendations.food.map(
                  (food, idx) => (
                    <div
                      key={idx}
                      className="p-5 rounded-xl border border-[#c0b3a5] shadow-md hover:shadow-xl transition-all duration-300 transform hover:scale-105 bg-gradient-to-br from-[#e1d5c4] to-white"
                    >
                      <h4 className="font-medium text-[#1e1b19] font-serif">
                        {food}
                      </h4>
                    </div>
                  )
                )}
              </div>
            </section>

            {/* Lifestyle Tips */}
            <section>
              <h3 className="text-2xl font-serif font-semibold text-[#1e1b19] mb-5 flex items-center">
                <span className="mr-3 bg-[#728394] text-white p-2 rounded-full">
                  💡
                </span>
                Lifestyle Tips
              </h3>
              <div className="space-y-4 bg-white rounded-2xl p-6 border border-[#c0b3a5] shadow-lg">
                {resultsData.gemini_response.recommendations.lifestyleTips.map(
                  (tip, idx) => (
                    <div
                      key={idx}
                      className="text-[#1e1b19] p-4 border-l-4 border-[#728394] bg-[#e1d5c4] bg-opacity-30 rounded-r-lg font-serif"
                    >
                      {tip}
                    </div>
                  )
                )}
              </div>
            </section>

            {/* Suggested Products */}
            <section>
              <h3 className="text-2xl font-serif font-semibold text-[#1e1b19] mb-5 flex items-center">
                <span className="mr-3 bg-[#728394] text-white p-2 rounded-full">
                  🧴
                </span>
                Suggested Products
              </h3>
              <div className="space-y-4 bg-white rounded-2xl p-6 border border-[#c0b3a5] shadow-lg">
                {resultsData.gemini_response.recommendations.suggestedProducts.map(
                  (product, idx) => (
                    <div
                      key={idx}
                      className="text-[#1e1b19] p-4 border-l-4 border-[#c0b3a5] bg-[#e1d5c4] bg-opacity-20 rounded-r-lg font-serif"
                    >
                      {product}
                    </div>
                  )
                )}
              </div>
            </section>

            {/* Save Results Button */}
            <div className="pt-6 flex justify-center">
              <button
                onClick={saveResults}
                className={`${
                  saved
                    ? "bg-[#728394]"
                    : "bg-gradient-to-r from-[#6d4f3e] to-[#829bab] hover:from-[#1e1b19] hover:to-[#728394]"
                } text-white py-3 px-8 rounded-full shadow-xl transform transition-all duration-300 font-serif text-lg flex items-center`}
              >
                {saved ? (
                  <>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5 mr-2"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    Saved!
                  </>
                ) : (
                  <>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5 mr-2"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path d="M7.707 10.293a1 1 0 10-1.414 1.414l3 3a1 1 0 001.414 0l3-3a1 1 0 00-1.414-1.414L11 11.586V6h5a2 2 0 012 2v7a2 2 0 01-2 2H4a2 2 0 01-2-2V8a2 2 0 012-2h5v5.586l-1.293-1.293zM9 4a1 1 0 012 0v2H9V4z" />
                    </svg>
                    Save Results
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        <div className="mt-8 text-center text-[#1e1b19] opacity-70 font-serif">
          © 2025 Skin Analysis System • All Rights Reserved
        </div>
      </div>
    </div>
  );
}

export default ResultsPage;
