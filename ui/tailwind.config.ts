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
      gridTemplateColumns: {
        '13': 'repeat(13, minmax(0, 1fr))',
        '14': 'repeat(14, minmax(0, 1fr))',
        '15': 'repeat(15, minmax(0, 1fr))',
        '16': 'repeat(16, minmax(0, 1fr))',
        '17': 'repeat(17, minmax(0, 1fr))',
        '18': 'repeat(18, minmax(0, 1fr))',
        '19': 'repeat(19, minmax(0, 1fr))',
        '20': 'repeat(20, minmax(0, 1fr))',
        '21': 'repeat(21, minmax(0, 1fr))',
        '22': 'repeat(22, minmax(0, 1fr))',
        '23': 'repeat(23, minmax(0, 1fr))',
        '24': 'repeat(24, minmax(0, 1fr))',
        '25': 'repeat(25, minmax(0, 1fr))',
        '26': 'repeat(26, minmax(0, 1fr))',
        '27': 'repeat(27, minmax(0, 1fr))',
        '28': 'repeat(28, minmax(0, 1fr))',
        '29': 'repeat(29, minmax(0, 1fr))',
        '30': 'repeat(30, minmax(0, 1fr))',
        '31': 'repeat(31, minmax(0, 1fr))',
        '32': 'repeat(32, minmax(0, 1fr))',
        '33': 'repeat(33, minmax(0, 1fr))',
        '34': 'repeat(34, minmax(0, 1fr))',
        '35': 'repeat(35, minmax(0, 1fr))',
        '36': 'repeat(36, minmax(0, 1fr))',
        '37': 'repeat(37, minmax(0, 1fr))',
        '38': 'repeat(38, minmax(0, 1fr))',
        '39': 'repeat(39, minmax(0, 1fr))',
        '40': 'repeat(40, minmax(0, 1fr))',
      },
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
