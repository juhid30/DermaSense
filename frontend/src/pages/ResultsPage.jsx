import React, { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";

function ResultsPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [saved, setSaved] = useState(false);
  const backgroundRef = useRef(null);

  // Get current date and time for timestamping
  const currentDate = new Date().toLocaleString();

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

    backgroundRef.current.style.background = `radial-gradient(circle at ${xPercent}% ${yPercent}%, #e6ccb2, #ddb892, #b08968)`;
  };

  return (
    <div
      ref={backgroundRef}
      onMouseMove={handleMouseMove}
      className="min-h-screen py-12 px-4 sm:px-6 transition-all duration-500 ease-in-out"
      style={{
        background:
          "radial-gradient(circle at center, #e6ccb2, #ddb892, #b08968)",
      }}
    >
      <div className="max-w-4xl mx-auto">
        <div className="bg-white bg-opacity-90 backdrop-blur-sm rounded-3xl shadow-2xl overflow-hidden border border-tan-400">
          <div className="bg-gradient-to-r from-coffee-700 to-raw_umber-600 py-6 px-8 flex justify-between items-center rounded-t-2xl">
            <h2 className="text-3xl font-serif font-bold text-stone-50 tracking-wider">
              Skin Analysis Results
            </h2>
            <div className="flex space-x-4">
              <button
                onClick={() => navigate("/camera")}
                className="text-stone-50 hover:text-coffee-200 transition-colors transform hover:scale-105"
              >
                New Analysis
              </button>
              <button
                onClick={() => navigate("/dashboard")}
                className="text-stone-50 hover:text-coffee-200 transition-colors transform hover:scale-105 flex items-center"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 mr-1"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z" />
                </svg>
                Dashboard
              </button>
            </div>
          </div>

          <div className="p-8 space-y-8">
            {/* Analysis Date/Time */}
            <div className="text-sm text-coffee-600 font-medium">
              Analysis performed: {currentDate}
            </div>

            {/* Prediction Results */}
            <section>
              <h3 className="text-xl font-serif font-semibold text-coffee-700 mb-3 flex items-center">
                <span className="mr-2">🧪</span> Prediction Results
              </h3>
              <div className="p-5 rounded-xl border border-coffee-200 shadow-sm hover:shadow-lg transition-all duration-300">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="p-4 bg-white bg-opacity-70 rounded-lg shadow-sm">
                    <p className="font-medium text-coffee-700 mb-2">
                      Detected Condition:
                    </p>
                    <p className="text-coffee-700 text-lg">
                      {resultsData.predicted_label_disease}
                    </p>
                  </div>

                  <div className="p-4 bg-white bg-opacity-70 rounded-lg shadow-sm">
                    <p className="font-medium text-coffee-700 mb-2">
                      Identified Concerns:
                    </p>
                    <ul className="list-disc ml-6 mt-1">
                      <p className="text-coffee-700 text-lg">
                        {resultsData.predicted_label_ageing}
                      </p>
                    </ul>
                  </div>
                </div>
              </div>
            </section>

            {/* AI Response */}
            <section>
              <h3 className="text-xl font-serif font-semibold text-coffee-700 mb-3 flex items-center">
                <span className="mr-2">🤖</span> Analysis Summary
              </h3>
              <div className="p-5 rounded-xl border border-coffee-200 shadow-sm hover:shadow-lg transition-all duration-300">
                <p className="text-coffee-700 leading-relaxed">
                  {resultsData.gemini_response.summary ||
                    "No analysis summary available."}
                </p>
              </div>
            </section>

            {/* Recommended Ingredients */}
            <section>
              <h3 className="text-xl font-serif font-semibold text-coffee-700 mb-3 flex items-center">
                <span className="mr-2">🌿</span> Recommended Active Ingredients
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {resultsData.gemini_response.recommendations.coreIngredients.map(
                  (ingredient, idx) => (
                    <div
                      key={idx}
                      className="p-4 rounded-xl border border-coffee-200 shadow-sm hover:shadow-lg transition-all duration-300 transform hover:scale-105"
                    >
                      <h4 className="font-medium text-coffee-700">
                        {ingredient}
                      </h4>
                    </div>
                  )
                )}
              </div>
            </section>

            {/* Recommended Foods */}
            <section>
              <h3 className="text-xl font-serif font-semibold text-coffee-700 mb-3 flex items-center">
                <span className="mr-2">🍎</span> Recommended Foods
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {resultsData.gemini_response.recommendations.food.map(
                  (food, idx) => (
                    <div
                      key={idx}
                      className="p-4 rounded-xl border border-coffee-200 shadow-sm hover:shadow-lg transition-all duration-300 transform hover:scale-105"
                    >
                      <h4 className="font-medium text-coffee-700">{food}</h4>
                    </div>
                  )
                )}
              </div>
            </section>

            {/* Lifestyle Tips */}
            <section>
              <h3 className="text-xl font-serif font-semibold text-coffee-700 mb-3 flex items-center">
                <span className="mr-2">💡</span> Lifestyle Tips
              </h3>
              <div className="space-y-3">
                {resultsData.gemini_response.recommendations.lifestyleTips.map(
                  (tip, idx) => (
                    <div key={idx} className="text-coffee-700">
                      {tip}
                    </div>
                  )
                )}
              </div>
            </section>

            {/* Suggested Products */}
            <section>
              <h3 className="text-xl font-serif font-semibold text-coffee-700 mb-3 flex items-center">
                <span className="mr-2">🧴</span> Suggested Products
              </h3>
              <div className="space-y-3">
                {resultsData.gemini_response.recommendations.suggestedProducts.map(
                  (product, idx) => (
                    <div key={idx} className="text-coffee-700">
                      {product}
                    </div>
                  )
                )}
              </div>
            </section>

            {/* Save Results Button */}
            <div className="pt-4 flex justify-center">
              <button
                onClick={saveResults}
                className={`${
                  saved
                    ? "bg-gradient-to-r from-tan-500 to-tan-600"
                    : "bg-gradient-to-r from-coffee-600 to-raw_umber-500 hover:from-coffee-700 hover:to-raw_umber-600"
                } text-white py-2 px-4 rounded-full shadow-xl transform transition-all duration-300`}
              >
                {saved ? "Saved!" : "Save Results"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResultsPage;
