/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // earthy / greenhouse palette
        moss: {
          50: '#f3f8f2',
          100: '#e2efe0',
          200: '#c5dfc2',
          300: '#9cc79a',
          400: '#6ea86d',
          500: '#4c8b4b',
          600: '#386f38',
          700: '#2c5830',
          800: '#24462a',
          900: '#1d3724',
        },
        clay: {
          50: '#fbf6f2',
          100: '#f5e9de',
          200: '#ead0b8',
          300: '#dcb08b',
          400: '#cd8a5c',
          500: '#c26f3a',
          600: '#b1572c',
          700: '#904426',
          800: '#753723',
          900: '#5f2e21',
        },
        soil: '#4a3f35',
        berry: '#c2415a',
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'scale-in': {
          '0%': { opacity: '0', transform: 'scale(.96)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        'fade-up': 'fade-up .35s ease-out both',
        'scale-in': 'scale-in .2s ease-out both',
      },
    },
  },
  plugins: [],
}
