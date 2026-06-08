import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        gray: {
          950: 'rgb(var(--gray-950) / <alpha-value>)',
          900: 'rgb(var(--gray-900) / <alpha-value>)',
          800: 'rgb(var(--gray-800) / <alpha-value>)',
          700: 'rgb(var(--gray-700) / <alpha-value>)',
          600: 'rgb(var(--gray-600) / <alpha-value>)',
          500: 'rgb(var(--gray-500) / <alpha-value>)',
          400: 'rgb(var(--gray-400) / <alpha-value>)',
          300: 'rgb(var(--gray-300) / <alpha-value>)',
          200: 'rgb(var(--gray-200) / <alpha-value>)',
          100: 'rgb(var(--gray-100) / <alpha-value>)',
        },

        yellow: {
          950: '#1a1203',
          900: '#2a1c05',
          800: '#3a2607',
          700: '#5a3a0b',
          600: '#7a4e0f',
          500: '#f59e0b', // deeper than base; good for hover on dark
          400: '#fbbf24', // your base
          300: '#fcd34d',
          200: '#fde68a',
          100: '#fef3c7',
          50: '#fffbeb',
        },
      },
    },
  },
  plugins: [],
};

export default config;
