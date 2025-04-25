import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";

function DashboardPage() {
  const navigate = useNavigate();
  const [skinProfile, setSkinProfile] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [clearConfirm, setClearConfirm] = useState(false);
  const location = useLocation();

  useEffect(() => {
    // Fetch skin profile and analysis history from localStorage
    const fetchData = () => {
      try {
        // Get skin profile
        const profileData = localStorage.getItem("skinProfile");
        if (profileData) {
          setSkinProfile(JSON.parse(profileData));
        }

        // Get analysis history
        const historyData = localStorage.getItem("skinAnalysisResults");
        if (historyData) {
          const parsedHistory = JSON.parse(historyData);
          setAnalysisHistory(
            Array.isArray(parsedHistory) ? parsedHistory : [parsedHistory]
          );
        } else {
          // Sample data if no history exists
          setAnalysisHistory([]);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Format date for better display
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return isNaN(date)
      ? dateString
      : date.toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        });
  };

  // Handle clicking on a history item to view full results
  const viewResult = (result) => {
    console.log("Viewing result:", result);
    navigate("/results", { state: { results: result } });
  };

  // Clear history
  const clearHistory = () => {
    if (clearConfirm) {
      localStorage.removeItem("skinAnalysisResults");
      setAnalysisHistory([]);
      setClearConfirm(false);
    } else {
      setClearConfirm(true);
      setTimeout(() => setClearConfirm(false), 3000); // Reset after 3 seconds
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen w-screen bg-gradient-to-r from-[#c0b3a5] to-[#829bab]">
        <div className="w-16 h-16 border-4 border-t-[#1e1b19] border-b-[#1e1b19] border-l-transparent border-r-transparent rounded-full animate-spin mb-6"></div>
        <div className="text-2xl text-[#1e1b19] font-medium animate-pulse">
          Loading your dashboard...
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#c0b3a5] to-[#829bab] p-6">
      <div className="max-w-6xl mx-auto">
        {/* Dashboard Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Header - Full Width */}
          <div className="lg:col-span-12">
            <div className="bg-[#1e1b19] rounded-2xl shadow-xl overflow-hidden border border-[#6d4f3e]">
              <div className="py-5 px-8 flex justify-between items-center">
                <h2 className="text-3xl font-serif font-light text-[#c0b3a5] tracking-wider">
                  MY SKIN DASHBOARD
                </h2>
                <button
                  onClick={() => navigate("/")}
                  className="text-[#829bab] hover:text-white transition-colors px-4 py-2 rounded-full border border-[#829bab] hover:border-white text-sm font-medium"
                >
                  Back to Home
                </button>
              </div>
            </div>
          </div>

          {/* Main Content Area - Combined Profile and Actions */}
          <div className="lg:col-span-12">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Profile Panel - Left Side */}
              <div className="lg:col-span-8">
                <div className="bg-[#f5f3f0] rounded-2xl shadow-xl overflow-hidden border-l-4 border-l-[#6d4f3e] h-full">
                  <div className="p-6">
                    <h3 className="text-2xl font-light text-[#1e1b19] mb-5 border-b border-[#c0b3a5] pb-2">
                      YOUR SKIN PROFILE
                    </h3>
                    <div className="bg-[#e9e5e1] p-6 rounded-lg shadow-inner">
                      {skinProfile ? (
                        <div className="grid grid-cols-2 gap-5">
                          <div className="p-5 bg-white rounded-lg shadow-md transition-transform hover:transform hover:scale-102">
                            <p className="font-light text-[#6d4f3e] mb-2 uppercase text-xs tracking-wider">
                              Skin Type
                            </p>
                            <p className="text-[#1e1b19] text-xl font-light">
                              {skinProfile.skinType || "Not specified"}
                            </p>
                          </div>
                          <div className="p-5 bg-white rounded-lg shadow-md transition-transform hover:transform hover:scale-102">
                            <p className="font-light text-[#6d4f3e] mb-2 uppercase text-xs tracking-wider">
                              Sensitivity
                            </p>
                            <p className="text-[#1e1b19] text-xl font-light">
                              {skinProfile.sensitivityLevel || "Not specified"}
                            </p>
                          </div>
                          <div className="p-5 bg-white rounded-lg shadow-md transition-transform hover:transform hover:scale-102">
                            <p className="font-light text-[#6d4f3e] mb-2 uppercase text-xs tracking-wider">
                              Primary Concerns
                            </p>
                            <p className="text-[#1e1b19] text-xl font-light">
                              {skinProfile.skinConcerns &&
                              skinProfile.skinConcerns.length > 0
                                ? skinProfile.skinConcerns.join(", ")
                                : "None specified"}
                            </p>
                          </div>
                          <div className="p-5 bg-white rounded-lg shadow-md transition-transform hover:transform hover:scale-102">
                            <p className="font-light text-[#6d4f3e] mb-2 uppercase text-xs tracking-wider">
                              Breakout Frequency
                            </p>
                            <p className="text-[#1e1b19] text-xl font-light">
                              {skinProfile.breakoutFrequency || "Not specified"}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center py-8 px-6 bg-white rounded-lg shadow-md">
                          <p className="text-[#1e1b19] font-light text-lg mb-4">
                            No skin profile found. Please take the quiz.
                          </p>
                          <button
                            onClick={() => navigate("/quiz")}
                            className="mt-2 bg-[#6d4f3e] hover:bg-[#1e1b19] text-white font-light py-3 px-8 rounded-full transition-all duration-300 transform hover:scale-105 shadow-lg text-sm uppercase tracking-wider"
                          >
                            Take Skin Quiz
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Tips Section - Lighter background with darker text */}
                    <div className="mt-6 p-5 bg-[#829bab] bg-opacity-20 rounded-lg border-l-4 border-l-[#829bab]">
                      <h3 className="text-lg font-medium mb-2 flex items-center text-[#1e1b19]">
                        <span className="mr-2">💡</span> Skin Care Tips
                      </h3>
                      <p className="text-[#1e1b19] font-light leading-relaxed">
                        Regular analysis helps track your skin's progress. We
                        recommend taking photos in similar lighting conditions
                        every 2-4 weeks for the most accurate comparisons and
                        personalized recommendations.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Panel - Right Side */}
              <div className="lg:col-span-4">
                <div className="bg-[#1e1b19] rounded-2xl shadow-xl overflow-hidden h-full">
                  <div className="p-6 flex flex-col h-full">
                    <h3 className="text-2xl font-light text-[#c0b3a5] mb-6 border-b border-[#6d4f3e] pb-2">
                      QUICK ACTIONS
                    </h3>

                    <div className="space-y-6 flex-grow flex flex-col justify-center">
                      <button
                        onClick={() => navigate("/camera")}
                        className="w-full bg-[#829bab] hover:bg-[#728394] text-white font-light py-4 px-6 rounded-xl text-lg transition-all duration-300 transform hover:scale-105 shadow-lg flex items-center justify-center"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-6 w-6 mr-3"
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
                        NEW ANALYSIS
                      </button>

                      <button
                        onClick={() => navigate("/quiz")}
                        className="w-full bg-[#c0b3a5] hover:bg-[#a99d8f] text-[#1e1b19] font-light py-4 px-6 rounded-xl text-lg transition-all duration-300 transform hover:scale-105 shadow-lg flex items-center justify-center"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-6 w-6 mr-3"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h14a2 2 0 002-2V7a2 2 0 00-2-2h-2m-6 0H5m7 4h3"
                          />
                        </svg>
                        UPDATE PROFILE
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* History Section */}
          <div className="lg:col-span-12">
            <div className="bg-[#f5f3f0] rounded-2xl shadow-xl overflow-hidden border-l-4 border-l-[#829bab]">
              <div className="p-6">
                <div className="flex justify-between items-center mb-5 border-b border-[#c0b3a5] pb-2">
                  <h3 className="text-2xl font-light text-[#1e1b19]">
                    ANALYSIS HISTORY
                  </h3>

                  {/* Clear History Button */}
                  {analysisHistory.length > 0 && (
                    <button
                      onClick={clearHistory}
                      className={`flex items-center px-4 py-2 rounded-full text-sm font-light tracking-wider transition-all duration-300 ${
                        clearConfirm
                          ? "bg-red-500 text-white"
                          : "border border-[#1e1b19] text-[#1e1b19] hover:bg-[#1e1b19] hover:text-white"
                      }`}
                    >
                      {clearConfirm ? (
                        <>
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-4 w-4 mr-2"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            />
                          </svg>
                          CONFIRM CLEAR
                        </>
                      ) : (
                        <>
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-4 w-4 mr-2"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M6 18L18 6M6 6l12 12"
                            />
                          </svg>
                          CLEAR HISTORY
                        </>
                      )}
                    </button>
                  )}
                </div>

                {analysisHistory.length === 0 ? (
                  <div className="text-center py-10 bg-white rounded-lg shadow-inner">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-12 w-12 mx-auto text-[#829bab] opacity-50 mb-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1}
                        d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                      />
                    </svg>
                    <p className="text-[#1e1b19] font-light text-lg">
                      No analysis history found.
                    </p>
                    <p className="text-[#6d4f3e] text-sm mt-2">
                      Start a new analysis to see results here.
                    </p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-4 mt-4">
                    {analysisHistory.map((result) => (
                      <div
                        key={result.id}
                        className="flex justify-between items-center p-5 bg-white border-l-4 border-l-[#829bab] rounded-lg shadow-md hover:shadow-lg transition duration-300 cursor-pointer"
                        onClick={() => viewResult(result)}
                      >
                        <div>
                          <p className="text-[#1e1b19] font-medium text-lg mb-1">
                            <span className="font-semibold">Acne:</span>{" "}
                            {result.predicted_label_disease.severity}
                            <span className="mx-2">|</span>
                            <span className="font-semibold">
                              Other Symptoms:
                            </span>{" "}
                            {result.predicted_label_ageing
                              .charAt(0)
                              .toUpperCase() +
                              result.predicted_label_ageing.slice(1)}
                          </p>
                          <p className="text-[#6d4f3e] font-light">
                            {result.gemini_response.summary}
                          </p>
                        </div>
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-6 w-6 text-[#829bab]"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 5l7 7-7 7"
                          />
                        </svg>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DashboardPage;
