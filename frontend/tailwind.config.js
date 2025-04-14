/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        almond: {
          DEFAULT: "#ede0d4",
          100: "#3f2c1a",
          200: "#7f5935",
          300: "#b88555",
          400: "#d2b295",
          500: "#ede0d4",
          600: "#f1e6dc",
          700: "#f4ece5",
          800: "#f8f3ee",
          900: "#fbf9f6",
        },
        dun: {
          DEFAULT: "#e6ccb2",
          100: "#3e2914",
          200: "#7b5228",
          300: "#b97a3c",
          400: "#d2a374",
          500: "#e6ccb2",
          600: "#ebd6c1",
          700: "#f0e0d1",
          800: "#f5ebe0",
          900: "#faf5f0",
        },
        tan: {
          DEFAULT: "#ddb892",
          100: "#382512",
          200: "#704923",
          300: "#a76e35",
          400: "#cb935b",
          500: "#ddb892",
          600: "#e4c6a8",
          700: "#ead4be",
          800: "#f1e2d4",
          900: "#f8f1e9",
        },
        chamoisee: {
          DEFAULT: "#b08968",
          100: "#251b13",
          200: "#493627",
          300: "#6e523a",
          400: "#936d4d",
          500: "#b08968",
          600: "#c0a087",
          700: "#cfb8a5",
          800: "#dfd0c3",
          900: "#efe7e1",
        },
        coffee: {
          DEFAULT: "#7f5539",
          100: "#19110b",
          200: "#332217",
          300: "#4c3322",
          400: "#65442e",
          500: "#7f5539",
          600: "#ac734d",
          700: "#c29678",
          800: "#d7b9a5",
          900: "#ebdcd2",
        },
        raw_umber: {
          DEFAULT: "#9c6644",
          100: "#1f140e",
          200: "#3e291b",
          300: "#5e3d29",
          400: "#7d5237",
          500: "#9c6644",
          600: "#b98260",
          700: "#cba288",
          800: "#dcc1b0",
          900: "#eee0d7",
        },
      },
      animation: {
        flash: "flash 0.5s ease-in-out",
      },
      keyframes: {
        flash: {
          "0%": { borderColor: "#fbbf24" }, // flash color
          "50%": { borderColor: "#f97316" }, // flash color
          "100%": { borderColor: "#fbbf24" }, // flash color
        },
      },
    },
  },
  plugins: [],
};
