import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

function QuizPage() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    skinType: "",
    skinConcerns: [],
    endOfDayFeel: "",
    breakoutFrequency: "",
    productsUsed: [],
    sensitivityLevel: "",
  });
  const [submitted, setSubmitted] = useState(false);

  const updateField = (field, value) =>
    setFormData((prev) => ({ ...prev, [field]: value }));
  const toggleArrayItem = (field, value) => {
    setFormData((prev) => {
      const list = prev[field];
      return {
        ...prev,
        [field]: list.includes(value)
          ? list.filter((i) => i !== value)
          : [...list, value],
      };
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("User Skin Information:", formData);

    // Save the form data to localStorage for the dashboard to use
    localStorage.setItem("skinProfile", JSON.stringify(formData));

    /* 
    // This would be used when connected to the backend
    const saveProfileToBackend = async () => {
      try {
        const response = await fetch("/api/save-profile", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            // Add authentication headers as needed
          },
          body: JSON.stringify(formData)
        });
  
        if (!response.ok) {
          throw new Error("Failed to save profile to server");
        }
      } catch (error) {
        console.error("Error saving profile to backend:", error);
        // Still save to localStorage as backup
        localStorage.setItem('skinProfile', JSON.stringify(formData));
      }
    };
  
    saveProfileToBackend();
    */

    setSubmitted(true);
  };

  if (submitted) {
    return (
      <div className="flex flex-col items-center justify-center h-screen w-screen bg-gradient-to-r from-almond-400 to-dun-300">
        {/* Thank You Icon */}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-16 w-16 text-coffee-600 mb-4"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M5 13l4 4L19 7"
          />
        </svg>

        {/* Thank You Text */}
        <div className="text-3xl font-semibold text-coffee-600 mb-2 animate-pulse">
          Thank You!
        </div>

        <p className="text-xl text-coffee-500 mb-10 animate-pulse">
          Your skin profile has been saved successfully.
        </p>

        {/* Button */}
        <button
          onClick={() => navigate("/dashboard")}
          className="bg-coffee-600 hover:bg-coffee-700 text-amber-50 font-normal py-3 px-10 rounded-full text-lg transition-all duration-300 shadow-md"
        >
          View Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-r from-tan-500 via-dun-400 to-almond-500 py-16 px-4 sm:px-6">
      <div className="max-w-3xl mx-auto bg-white rounded-3xl shadow-xl overflow-hidden border border-dun-400 p-6">
        <div className="bg-coffee-600 py-6 px-8 flex items-center rounded-t-3xl mb-10">
          <h2 className="text-3xl font-semibold text-white tracking-wide">
            Skin Profile Quiz
          </h2>
          <span className="ml-2 text-dun-200 text-2xl">✧</span>
        </div>

        <form onSubmit={handleSubmit} className="space-y-12">
          <div className="space-y-10">
            <Question
              label="1. What is your skin type?"
              type="radio"
              options={["Oily", "Dry", "Combination", "Normal", "Sensitive"]}
              selected={formData.skinType}
              onChange={(value) => updateField("skinType", value)}
            />
            <Question
              label="2. What are your primary skin concerns? (Choose all that apply)"
              type="checkbox"
              options={[
                "Acne or pimples",
                "Blackheads or whiteheads",
                "Dark spots or hyperpigmentation",
                "Fine lines or wrinkles",
                "Redness or irritation",
                "Uneven skin tone",
                "Large pores",
                "Dry patches",
              ]}
              selected={formData.skinConcerns}
              onChange={(value) => toggleArrayItem("skinConcerns", value)}
            />
            <Question
              label="3. How does your skin usually feel at the end of the day?"
              type="radio"
              options={[
                "Very oily and shiny",
                "A bit oily in the T-zone",
                "Balanced",
                "Tight and dry",
                "Itchy or inflamed",
              ]}
              selected={formData.endOfDayFeel}
              onChange={(value) => updateField("endOfDayFeel", value)}
            />
            <Question
              label="4. How often do you experience breakouts?"
              type="radio"
              options={[
                "Almost never",
                "Occasionally (a few times a year)",
                "Frequently (once a month or more)",
                "Constantly",
              ]}
              selected={formData.breakoutFrequency}
              onChange={(value) => updateField("breakoutFrequency", value)}
            />
            <Question
              label="5. What kind of products do you usually use? (Choose all that apply)"
              type="checkbox"
              options={[
                "Cleanser",
                "Toner",
                "Moisturizer",
                "Sunscreen",
                "Serum",
                "Face oils",
                "Exfoliators (scrubs or acids)",
                "Face masks",
              ]}
              selected={formData.productsUsed}
              onChange={(value) => toggleArrayItem("productsUsed", value)}
            />
            <Question
              label="6. What is your skin's sensitivity level?"
              type="radio"
              options={[
                "Not sensitive at all",
                "Slightly sensitive (occasional irritation)",
                "Very sensitive (reacts to many products)",
              ]}
              selected={formData.sensitivityLevel}
              onChange={(value) => updateField("sensitivityLevel", value)}
            />
          </div>

          <div className="mt-6 flex justify-center">
            <button
              type="submit"
              className="w-full bg-coffee-600 hover:bg-coffee-700 text-amber-50 font-medium py-4 px-6 rounded-full shadow-lg transition duration-200 transform hover:scale-105"
            >
              Submit
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function Question({ label, options, selected, onChange, type }) {
  return (
    <div className="p-6 bg-white rounded-xl shadow-md space-y-6 border border-tan-200">
      <h3 className="text-lg font-semibold text-coffee-700 mb-4">{label}</h3>
      <div
        className={
          type === "checkbox"
            ? "grid grid-cols-1 sm:grid-cols-2 gap-6"
            : "space-y-4"
        }
      >
        {options.map((option) => (
          <div key={option} className="flex items-center">
            <input
              type={type}
              id={`${label}-${option}`}
              name={label}
              value={option}
              checked={
                type === "checkbox"
                  ? selected.includes(option)
                  : selected === option
              }
              onChange={() => onChange(option)}
              className="h-5 w-5 text-coffee-500 focus:ring-coffee-400"
              required={type !== "checkbox"}
            />
            <label
              htmlFor={`${label}-${option}`}
              className="ml-2 text-amber-900"
            >
              {option}
            </label>
          </div>
        ))}
      </div>
    </div>
  );
}

export default QuizPage;
