import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

function DashboardPage() {
  const navigate = useNavigate();
  const [skinProfile, setSkinProfile] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [clearConfirm, setClearConfirm] = useState(false);

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
  const viewResult = (resultId) => {
    // In a real implementation, you might navigate to the results page with this ID
    // For now we'll just console log it
    console.log("Viewing result:", resultId);
    navigate("/results");
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
      <div className="flex flex-col items-center justify-center h-screen w-screen bg-gradient-to-r from-tan-500 via-dun-400 to-almond-500">
        <div className="w-16 h-16 border-4 border-t-coffee-400 border-b-coffee-400 border-l-transparent border-r-transparent rounded-full animate-spin mb-6"></div>
        <div className="text-2xl text-coffee-800 font-medium animate-pulse">
          Loading your dashboard...
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-r from-tan-500 via-dun-400 to-almond-500 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Dashboard Grid Layout - Adjusted for less scrolling */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Header - Full Width */}
          <div className="lg:col-span-12">
            <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-dun-400">
              <div className="bg-coffee-600 py-4 px-6 flex justify-between items-center rounded-t-xl">
                <h2 className="text-2xl font-serif font-bold text-stone-50 tracking-wider">
                  My Skin Dashboard
                </h2>
                <button
                  onClick={() => navigate("/")}
                  className="text-stone-50 hover:text-coffee-200 transition-colors transform hover:scale-105"
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
                <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-dun-400 h-full">
                  <div className="p-5">
                    <h3 className="text-xl font-semibold text-coffee-600 mb-3">
                      Your Skin Profile
                    </h3>
                    <div className="bg-raw-umber-50 p-4 rounded-lg border border-coffee-200 shadow-sm">
                      {skinProfile ? (
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-3 bg-white rounded-lg shadow-sm">
                            <p className="font-medium text-coffee-600 mb-1">
                              Skin Type:
                            </p>
                            <p className="text-coffee-700 text-lg font-medium">
                              {skinProfile.skinType || "Not specified"}
                            </p>
                          </div>
                          <div className="p-3 bg-white rounded-lg shadow-sm">
                            <p className="font-medium text-coffee-600 mb-1">
                              Sensitivity:
                            </p>
                            <p className="text-coffee-700 text-lg font-medium">
                              {skinProfile.sensitivityLevel || "Not specified"}
                            </p>
                          </div>
                          <div className="p-3 bg-white rounded-lg shadow-sm">
                            <p className="font-medium text-coffee-600 mb-1">
                              Primary Concerns:
                            </p>
                            <p className="text-coffee-700 font-medium">
                              {skinProfile.skinConcerns &&
                              skinProfile.skinConcerns.length > 0
                                ? skinProfile.skinConcerns.join(", ")
                                : "None specified"}
                            </p>
                          </div>
                          <div className="p-3 bg-white rounded-lg shadow-sm">
                            <p className="font-medium text-coffee-600 mb-1">
                              Breakout Frequency:
                            </p>
                            <p className="text-coffee-700 text-lg font-medium">
                              {skinProfile.breakoutFrequency || "Not specified"}
                            </p>
                          </div>
                        </div>
                      ) : (
                        <div className="text-center py-4">
                          <p className="text-coffee-600 font-medium">
                            No skin profile found. Please take the quiz.
                          </p>
                          <button
                            onClick={() => navigate("/quiz")}
                            className="mt-4 bg-coffee-600 hover:bg-coffee-700 text-stone-50 font-medium py-2 px-6 rounded-full transition-all duration-300 transform hover:scale-105"
                          >
                            Take Skin Quiz
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Tips Section - Lighter background with darker text */}
                    <div className="mt-4 p-3 bg-tan-50 rounded-lg border border-coffee-200">
                      <h3 className="text-lg font-semibold mb-1 flex items-center text-coffee-600">
                        <span className="mr-2">💡</span> Skin Care Tips
                      </h3>
                      <p className="text-coffee-700 font-medium">
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
                <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-dun-400 h-full">
                  <div className="p-5 flex flex-col h-full">
                    <h3 className="text-xl font-semibold text-coffee-600 mb-4">
                      Quick Actions
                    </h3>

                    <div className="space-y-4 flex-grow flex flex-col justify-center">
                      <button
                        onClick={() => navigate("/camera")}
                        className="w-full bg-coffee-600 hover:bg-coffee-700 text-stone-50 font-bold py-3 px-4 rounded-xl text-lg transition-all duration-300 transform hover:scale-105 shadow-md flex items-center justify-center"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-6 w-6 mr-2"
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
                        Start New Analysis
                      </button>

                      <button
                        onClick={() => navigate("/quiz")}
                        className="w-full bg-tan-500 hover:bg-tan-600 text-stone-50 font-bold py-3 px-4 rounded-xl text-lg transition-all duration-300 transform hover:scale-105 shadow-md flex items-center justify-center"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-6 w-6 mr-2"
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
                        Update Skin Profile
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* History Section with Clear Button */}
          <div className="lg:col-span-12">
            <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-dun-400">
              <div className="p-5">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-semibold text-coffee-600">
                    Analysis History
                  </h3>

                  {/* Sophisticated Clear History Button */}
                  {analysisHistory.length > 0 && (
                    <button
                      onClick={clearHistory}
                      className={`flex items-center px-3 py-1 rounded-lg text-sm font-medium transition-all duration-300 ${
                        clearConfirm
                          ? "bg-red-500 text-white hover:bg-red-600"
                          : "bg-coffee-600 text-coffee-900 hover:bg-tan-200"
                      }`}
                    >
                      {clearConfirm ? (
                        <>
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-4 w-4 mr-1"
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
                          Confirm Clear
                        </>
                      ) : (
                        <>
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-4 w-4 mr-1"
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
                          Clear History
                        </>
                      )}
                    </button>
                  )}
                </div>

                {analysisHistory.length === 0 ? (
                  <div className="text-center py-4">
                    <p className="text-coffee-700 font-medium">
                      No history found.
                    </p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-3">
                    {analysisHistory.map((result) => (
                      <div
                        key={result.id}
                        className="flex justify-between items-center p-3 bg-raw-umber-50 border border-coffee-200 rounded-lg shadow-sm hover:bg-raw-umber-100 transition duration-200 cursor-pointer"
                        onClick={() => viewResult(result.id)}
                      >
                        <div>
                          <p className="text-coffee-600 font-medium">
                            {result.predicted_label_ageing}
                          </p>
                          {result.gemini_response.summary}
                          <p></p>
                        </div>
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
