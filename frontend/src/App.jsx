import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage.jsx";
import QuizPage from "./pages/QuizPage.jsx";
import LoadingPage from "./pages/LoadingPage";
import CameraPage from "./pages/CameraPage.jsx";
import ResultsPage from "./pages/ResultsPage.jsx";
import DashboardPage from "./pages/DashboardPage.jsx";

function App() {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // When the page refreshes, show loading for a short while
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 750); // Adjust time if needed

    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return <LoadingPage />;
  }

  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/quiz" element={<QuizPage />} />
        <Route path="/loading" element={<LoadingPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/camera" element={<CameraPage />} />
        <Route path="/results" element={<ResultsPage />} />
      </Routes>
    </Router>
  );
}

export default App;
